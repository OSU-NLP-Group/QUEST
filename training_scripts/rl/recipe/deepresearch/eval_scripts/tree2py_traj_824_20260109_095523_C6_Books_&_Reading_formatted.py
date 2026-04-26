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
TASK_ID = "booker_2024_winner_eval"
TASK_DESCRIPTION = """
Identify the fiction book that won the 2024 Booker Prize and verify it meets the following criteria: (1) the book has fewer than 250 pages in its UK hardcover first edition published by Jonathan Cape or Vintage UK; (2) the author is British and was born in the 1970s; (3) the book was also shortlisted or won at least one other major literary award in 2024 besides the Booker Prize. Provide the complete book title, author's full name, exact birth year, exact page count of the UK first edition, UK publisher name, publication month and year, and the name and status (winner or shortlisted) of the additional award. Additionally, identify the setting or primary theme of the novel based on descriptions from official sources or major book reviews.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BookerExtraction(BaseModel):
    """Structured extraction of the answer content."""
    # Winner identification
    book_title: Optional[str] = None
    booker_winner_urls: List[str] = Field(default_factory=list)

    # UK first edition and length criteria
    uk_publisher_name: Optional[str] = None
    uk_edition_type: Optional[str] = None  # e.g., "UK hardcover first edition"
    uk_hardcover_page_count: Optional[str] = None  # keep as string (e.g., "240")
    uk_first_edition_urls: List[str] = Field(default_factory=list)  # sources for publisher/edition/pages

    # Publication date requirement
    publication_month_year: Optional[str] = None  # e.g., "October 2024"
    publication_info_urls: List[str] = Field(default_factory=list)

    # Author criteria
    author_name: Optional[str] = None
    author_birth_year: Optional[str] = None
    author_bio_urls: List[str] = Field(default_factory=list)

    # Additional award requirement
    additional_award_name: Optional[str] = None  # e.g., "Costa Book Awards"
    additional_award_status: Optional[str] = None  # "winner" or "shortlisted"
    additional_award_year: Optional[str] = None  # e.g., "2024"
    additional_award_urls: List[str] = Field(default_factory=list)

    # Setting/Theme requirement
    setting_or_theme_desc: Optional[str] = None
    setting_or_theme_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_booker_fields() -> str:
    return """
    Extract the following fields exactly as they appear in the answer. If any field is missing, return null (for strings) or an empty array (for URLs). Extract URLs explicitly present in the answer; do not infer URLs.

    Required fields:
    1) book_title: The complete title of the book identified as the 2024 Booker Prize winner.
    2) booker_winner_urls: All URLs cited that support the claim that this book is the official winner of the 2024 Booker Prize (e.g., official Booker Prize page, BBC, Guardian, etc.). Return as a list.

    3) uk_publisher_name: The UK publisher name for the UK hardcover first edition (e.g., "Jonathan Cape" or "Vintage").
    4) uk_edition_type: The edition wording used in the answer (e.g., "UK hardcover first edition", "UK hardback first edition").
    5) uk_hardcover_page_count: The exact page count stated for the UK hardcover first edition (e.g., "240").
    6) uk_first_edition_urls: All URLs that support the UK hardcover first edition details (publisher, edition type, and page count). Return as a list.

    7) publication_month_year: The publication month and year of the UK hardcover first edition (e.g., "October 2024").
    8) publication_info_urls: All URLs that support the publication timing information. Return as a list.

    9) author_name: The author's full name.
    10) author_birth_year: The author's exact birth year (e.g., "1975").
    11) author_bio_urls: All URLs cited that support the author's bio (nationality/birth year). Return as a list.

    12) additional_award_name: The name of at least one other major literary award in 2024, not the Booker Prize.
    13) additional_award_status: The status for that award — must be "winner" or "shortlisted".
    14) additional_award_year: The award year (e.g., "2024").
    15) additional_award_urls: All URLs cited that support the additional award claim. Return as a list.

    16) setting_or_theme_desc: A brief description of the novel’s primary setting or theme as presented in the answer.
    17) setting_or_theme_urls: All URLs cited that support the setting/theme description from official sources or major reviews. Return as a list.

    Notes:
    - Do not invent information. Extract exactly from the answer text.
    - For URLs, include only valid URLs explicitly present in the answer (plain URLs or markdown links).
    - If a URL is missing a protocol, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_int_from_string(s: Optional[str]) -> Optional[int]:
    """Extract the first integer found in a string."""
    if not s:
        return None
    match = re.search(r"\d{1,4}", s)
    if match:
        try:
            return int(match.group(0))
        except Exception:
            return None
    return None


def is_allowed_publisher(name: Optional[str]) -> bool:
    """Check if publisher is Jonathan Cape or Vintage UK (allow common variants)."""
    if not name:
        return False
    normalized = name.strip().lower()
    if "jonathan cape" in normalized:
        return True
    # Accept "Vintage", "Vintage UK", "Vintage Books" for Vintage UK
    if "vintage" in normalized:
        return True
    return False


def contains_month_and_year(s: Optional[str]) -> bool:
    """Check the presence of a month name and a 4-digit year."""
    if not s:
        return False
    months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]
    lower = s.strip().lower()
    has_month = any(m in lower for m in months)
    has_year = re.search(r"\b(19|20)\d{2}\b", lower) is not None
    return has_month and has_year


def unique_urls(urls: List[str]) -> List[str]:
    """Return unique URLs preserving order."""
    seen = set()
    out = []
    for u in urls:
        if u and u not in seen:
            out.append(u)
            seen.add(u)
    return out


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_booker_prize_winner_identification(
    evaluator: Evaluator,
    parent_node,
    ex: BookerExtraction
) -> None:
    # Critical parallel node
    section = evaluator.add_parallel(
        id="Booker_Prize_Winner_Identification",
        desc="Correctly identify the 2024 Booker Prize winner and provide the complete title.",
        parent=parent_node,
        critical=True
    )

    # Complete book title provided (existence check)
    evaluator.add_custom_node(
        result=bool(ex.book_title and ex.book_title.strip()),
        id="Complete_Book_Title_Provided",
        desc="The complete book title is provided.",
        parent=section,
        critical=True
    )

    # The identified book is the official winner of the 2024 Booker Prize
    node_winner = evaluator.add_leaf(
        id="Book_Is_2024_Booker_Winner",
        desc="The identified book is the official winner of the 2024 Booker Prize.",
        parent=section,
        critical=True
    )
    claim = f"The book titled '{ex.book_title or 'UNKNOWN'}' is the official winner of the 2024 Booker Prize."
    await evaluator.verify(
        claim=claim,
        node=node_winner,
        sources=unique_urls(ex.booker_winner_urls),
        additional_instruction="Confirm the book is the winner (not just shortlisted) of the 2024 Booker Prize using the cited sources."
    )


async def verify_uk_first_edition_and_length(
    evaluator: Evaluator,
    parent_node,
    ex: BookerExtraction
) -> None:
    # Critical parallel node
    section = evaluator.add_parallel(
        id="UK_First_Edition_And_Length_Criteria",
        desc="Provide UK hardcover first-edition details and verify the <250 pages and publisher constraints.",
        parent=parent_node,
        critical=True
    )

    # UK publisher name provided
    evaluator.add_custom_node(
        result=bool(ex.uk_publisher_name and ex.uk_publisher_name.strip()),
        id="UK_Publisher_Name_Provided",
        desc="The UK publisher name is provided.",
        parent=section,
        critical=True
    )

    # UK publisher is allowed (Jonathan Cape or Vintage UK)
    evaluator.add_custom_node(
        result=is_allowed_publisher(ex.uk_publisher_name),
        id="UK_Publisher_Is_Allowed",
        desc="The UK hardcover first edition publisher is Jonathan Cape or Vintage UK.",
        parent=section,
        critical=True
    )

    # UK hardcover first edition stated (verify via sources)
    node_edition = evaluator.add_leaf(
        id="UK_Hardcover_First_Edition_Stated",
        desc="The edition referenced is stated to be the UK hardcover first edition.",
        parent=section,
        critical=True
    )
    claim_edition = (
        f"The cited sources indicate that the edition referenced is the UK hardcover first edition "
        f"(e.g., 'hardback'/'hardcover', 'first edition', UK publisher)."
    )
    await evaluator.verify(
        claim=claim_edition,
        node=node_edition,
        sources=unique_urls(ex.uk_first_edition_urls),
        additional_instruction="Look for edition indicators (UK hardback/hardcover, first edition) on official publisher pages or authoritative bibliographic sources."
    )

    # Exact page count provided (existence)
    evaluator.add_custom_node(
        result=bool(ex.uk_hardcover_page_count and ex.uk_hardcover_page_count.strip()),
        id="Exact_Page_Count_Provided",
        desc="The exact page count for the UK hardcover first edition is provided.",
        parent=section,
        critical=True
    )

    # Page count under 250 (numeric check)
    page_int = parse_int_from_string(ex.uk_hardcover_page_count)
    evaluator.add_custom_node(
        result=bool(page_int is not None and page_int < 250),
        id="Page_Count_Under_250",
        desc="The UK hardcover first edition page count is fewer than 250 pages.",
        parent=section,
        critical=True
    )


async def verify_publication_date_requirement(
    evaluator: Evaluator,
    parent_node,
    ex: BookerExtraction
) -> None:
    # Critical parallel node
    section = evaluator.add_parallel(
        id="Publication_Date_Requirement",
        desc="Provide publication timing information.",
        parent=parent_node,
        critical=True
    )

    # Publication month and year provided (format check)
    evaluator.add_custom_node(
        result=contains_month_and_year(ex.publication_month_year),
        id="Publication_Month_And_Year_Provided",
        desc="The publication month and year are provided.",
        parent=section,
        critical=True
    )


async def verify_author_criteria(
    evaluator: Evaluator,
    parent_node,
    ex: BookerExtraction
) -> None:
    # Critical parallel node
    section = evaluator.add_parallel(
        id="Author_Criteria",
        desc="Provide author identity and verify author constraints.",
        parent=parent_node,
        critical=True
    )

    # Author full name provided
    evaluator.add_custom_node(
        result=bool(ex.author_name and ex.author_name.strip()),
        id="Author_Full_Name_Provided",
        desc="The author's full name is provided.",
        parent=section,
        critical=True
    )

    # Author is British (verify via sources)
    node_british = evaluator.add_leaf(
        id="Author_Is_British",
        desc="The author is British (from the UK).",
        parent=section,
        critical=True
    )
    claim_british = f"The author {ex.author_name or 'UNKNOWN'} is British (from the UK)."
    await evaluator.verify(
        claim=claim_british,
        node=node_british,
        sources=unique_urls(ex.author_bio_urls),
        additional_instruction="Use reliable biographical sources (publisher bio, major media, Wikipedia infobox) to confirm UK nationality or being commonly recognized as British."
    )

    # Exact author birth year provided (existence)
    evaluator.add_custom_node(
        result=bool(ex.author_birth_year and ex.author_birth_year.strip()),
        id="Exact_Author_Birth_Year_Provided",
        desc="The author's exact birth year is provided.",
        parent=section,
        critical=True
    )

    # Author born in 1970s (verify via sources, do not rely solely on extracted year)
    node_1970s = evaluator.add_leaf(
        id="Author_Born_In_1970s",
        desc="The author's birth year is in the 1970s (1970–1979).",
        parent=section,
        critical=True
    )
    claim_1970s = f"{ex.author_name or 'UNKNOWN'} was born in the 1970s (between 1970 and 1979)."
    await evaluator.verify(
        claim=claim_1970s,
        node=node_1970s,
        sources=unique_urls(ex.author_bio_urls),
        additional_instruction="Confirm the birth year on biographical sources; accept if the year falls within 1970–1979 inclusive."
    )


async def verify_additional_award_requirement(
    evaluator: Evaluator,
    parent_node,
    ex: BookerExtraction
) -> None:
    # Critical parallel node
    section = evaluator.add_parallel(
        id="Additional_Award_2024_Requirement",
        desc="Provide and qualify at least one other major literary award in 2024 besides the Booker Prize.",
        parent=parent_node,
        critical=True
    )

    # Award name provided
    evaluator.add_custom_node(
        result=bool(ex.additional_award_name and ex.additional_award_name.strip()),
        id="Additional_Award_Name_Provided",
        desc="The name of at least one other major literary award in 2024 (besides the Booker Prize) is provided.",
        parent=section,
        critical=True
    )

    # Award status provided (winner or shortlisted)
    status_norm = (ex.additional_award_status or "").strip().lower()
    evaluator.add_custom_node(
        result=status_norm in {"winner", "shortlisted"},
        id="Additional_Award_Status_Provided",
        desc="The status for that additional award is specified as winner or shortlisted.",
        parent=section,
        critical=True
    )

    # Award is 2024 and not Booker (verify via sources)
    node_award_2024 = evaluator.add_leaf(
        id="Additional_Award_Is_2024_And_Not_Booker",
        desc="The additional award recognition is for 2024 and is not the Booker Prize itself.",
        parent=section,
        critical=True
    )
    claim_award = (
        f"In 2024, the book received recognition (status '{ex.additional_award_status or 'UNKNOWN'}') "
        f"for the '{ex.additional_award_name or 'UNKNOWN'}' award, which is not the Booker Prize."
    )
    await evaluator.verify(
        claim=claim_award,
        node=node_award_2024,
        sources=unique_urls(ex.additional_award_urls),
        additional_instruction="Verify the award year is 2024 and the award is not the Booker Prize. Use official award sites or trusted media coverage."
    )


async def verify_setting_or_theme_requirement(
    evaluator: Evaluator,
    parent_node,
    ex: BookerExtraction
) -> None:
    # Critical parallel node
    section = evaluator.add_parallel(
        id="Setting_Or_Theme_Requirement",
        desc="Describe the novel's primary setting or theme and ground it in official sources or major reviews (as requested).",
        parent=parent_node,
        critical=True
    )

    # Setting or theme described (existence)
    evaluator.add_custom_node(
        result=bool(ex.setting_or_theme_desc and ex.setting_or_theme_desc.strip()),
        id="Setting_Or_Theme_Described",
        desc="The primary setting or theme of the novel is described.",
        parent=section,
        critical=True
    )

    # Grounded in official sources or major reviews (verify via URLs)
    node_grounding = evaluator.add_leaf(
        id="Setting_Or_Theme_Grounded_In_Official_Or_Major_Review",
        desc="The setting/theme description is explicitly framed as based on official sources or major book reviews (e.g., attribution such as 'publisher description' / 'official site' / 'major review').",
        parent=section,
        critical=True
    )
    claim_grounding = (
        f"The described primary setting/theme—'{ex.setting_or_theme_desc or 'UNKNOWN'}'—is supported "
        f"by the cited official sources or major reviews."
    )
    await evaluator.verify(
        claim=claim_grounding,
        node=node_grounding,
        sources=unique_urls(ex.setting_or_theme_urls),
        additional_instruction="Focus on whether the gist of the described setting/theme matches what is stated in the official publisher description or major reviews (e.g., Guardian, NYT). Minor phrasing differences are acceptable."
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
    Evaluate an answer for the 2024 Booker Prize winner task.
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
        default_model=model,
    )

    # Extract structured information from the answer
    ex = await evaluator.extract(
        prompt=prompt_extract_booker_fields(),
        template_class=BookerExtraction,
        extraction_name="booker_2024_extraction",
    )

    # Build the critical root-level task node
    complete_task_node = evaluator.add_parallel(
        id="Complete_Booker_Prize_Task",
        desc="Identify the 2024 Booker Prize-winning fiction book and provide all required attributes while satisfying all stated constraints.",
        parent=root,
        critical=True
    )

    # Verification sections (all children must be critical since parent is critical)
    await verify_booker_prize_winner_identification(evaluator, complete_task_node, ex)
    await verify_uk_first_edition_and_length(evaluator, complete_task_node, ex)
    await verify_publication_date_requirement(evaluator, complete_task_node, ex)
    await verify_author_criteria(evaluator, complete_task_node, ex)
    await verify_additional_award_requirement(evaluator, complete_task_node, ex)
    await verify_setting_or_theme_requirement(evaluator, complete_task_node, ex)

    # Return structured result
    return evaluator.get_summary()