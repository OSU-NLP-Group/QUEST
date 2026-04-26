import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pulitzer_2025_fiction"
TASK_DESCRIPTION = """
For library cataloging purposes, identify the book that won the 2025 Pulitzer Prize for Fiction. Provide the following information with supporting URLs from authoritative sources:

1. The complete book title
2. The author's full name
3. The publisher's name
4. The complete publication date (month, day, and year)
5. A URL to an official source (such as pulitzer.org) confirming the Pulitzer Prize win
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PulitzerWinnerExtraction(BaseModel):
    """
    Structured extraction of the 2025 Pulitzer Prize for Fiction winner information.
    All fields must be extracted strictly from the provided answer text.
    """
    book_title: Optional[str] = None
    author_name: Optional[str] = None
    publisher_name: Optional[str] = None
    publication_date: Optional[str] = None  # Keep as string to maximize compatibility
    official_award_url: Optional[str] = None  # Prefer a pulitzer.org URL explicitly referenced for the award
    sources: List[str] = Field(default_factory=list)  # All URLs mentioned in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_winner_info() -> str:
    return """
    Extract, strictly from the answer text, the following fields about the 2025 Pulitzer Prize for Fiction winner:

    - book_title: The complete title of the winning book, exactly as written in the answer (include subtitle if present).
    - author_name: The author's full name, exactly as written in the answer.
    - publisher_name: The name of the publisher, exactly as written in the answer.
    - publication_date: The book’s publication date as written in the answer. Prefer a complete date including month, day, and year if provided (e.g., "April 16, 2024" or "2024-04-16").
    - official_award_url: The single URL from pulitzer.org in the answer that explicitly confirms the 2025 Pulitzer Prize for Fiction winner (e.g., a winner or prize announcement page). If multiple pulitzer.org URLs are present, pick the one that most directly confirms the award. If no pulitzer.org URL is present, set this field to null.
    - sources: An array of all URLs explicitly present in the answer text (including the official_award_url if present). Include every valid URL you find in the answer (plain URLs or markdown links).

    Rules:
    - Extract only what appears in the answer. Do not infer or invent.
    - For URLs, return absolute URLs with protocol; ignore malformed ones.
    - If a field is missing in the answer, set it to null (or [] for sources).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def pick_official_award_url(extracted: PulitzerWinnerExtraction) -> Optional[str]:
    """
    Choose the official award URL to use for verification:
    1) Prefer extracted.official_award_url if present.
    2) Else, search in sources for a pulitzer.org URL.
    """
    if extracted.official_award_url and "pulitzer.org" in extracted.official_award_url.lower():
        return extracted.official_award_url
    for u in extracted.sources:
        if isinstance(u, str) and "pulitzer.org" in u.lower():
            return u
    return extracted.official_award_url or None


def unique_urls(urls: List[Optional[str]]) -> List[str]:
    """Deduplicate while preserving order; drop falsy values."""
    seen = set()
    result: List[str] = []
    for u in urls:
        if not u:
            continue
        key = u.strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def is_complete_date(date_str: Optional[str]) -> bool:
    """
    Heuristic check whether the date string appears to include month, day, and year.
    Accepts:
    - "April 16, 2024" (month name + day + year)
    - "2024-04-16" (ISO)
    - "04/16/2024" or "4/16/2024" (US numeric)
    - "16 April 2024"
    """
    if not date_str:
        return False
    s = date_str.strip()
    if not s:
        return False

    month_names = r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    # Month name + day + year (order-insensitive for day/month)
    if re.search(rf"\b(?:{month_names})\b", s, flags=re.I) and re.search(r"\b\d{{1,2}}\b", s) and re.search(r"\b\d{{4}}\b", s):
        return True
    # ISO (YYYY-MM-DD or YYYY/M/D etc.)
    if re.search(r"\b\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}\b", s):
        return True
    # US numeric (M/D/YYYY or MM/DD/YYYY etc.)
    if re.search(r"\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}\b", s):
        return True
    return False


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_official_award_section(
    evaluator: Evaluator,
    parent_node,
    info: PulitzerWinnerExtraction,
) -> None:
    """
    Build and verify the 'Official_Award_Source_URL' section:
    - Check that an official pulitzer.org URL exists.
    - Verify that the page confirms the 2025 Pulitzer Prize for Fiction winner (title/author if available).
    """
    section = evaluator.add_parallel(
        id="Official_Award_Source_URL",
        desc="Provide a URL to an official or authoritative source (such as pulitzer.org) that confirms the 2025 Pulitzer Prize Fiction winner",
        parent=parent_node,
        critical=True,
    )

    used_award_url = pick_official_award_url(info)

    # Existence + domain check (critical)
    award_url_present_node = evaluator.add_custom_node(
        result=bool(used_award_url) and ("pulitzer.org" in used_award_url.lower()),
        id="official_award_url_present",
        desc="An official pulitzer.org URL confirming the winner is provided",
        parent=section,
        critical=True,
    )

    # Verification that the award page confirms the specific winner (critical)
    award_confirms_node = evaluator.add_leaf(
        id="official_award_page_confirms",
        desc="The pulitzer.org page confirms the 2025 Pulitzer Prize for Fiction winner",
        parent=section,
        critical=True,
    )

    # Craft the most specific claim possible given available fields
    title = info.book_title or ""
    author = info.author_name or ""
    if title and author:
        claim = f"The page confirms that the winner of the 2025 Pulitzer Prize for Fiction is the book '{title}' by {author}."
    elif title:
        claim = f"The page confirms that the winner of the 2025 Pulitzer Prize for Fiction is the book '{title}'."
    elif author:
        claim = f"The page confirms that the winner of the 2025 Pulitzer Prize for Fiction is written by {author}."
    else:
        claim = "The page confirms the winner of the 2025 Pulitzer Prize for Fiction."

    await evaluator.verify(
        claim=claim,
        node=award_confirms_node,
        sources=used_award_url,
        additional_instruction="Focus only on whether this pulitzer.org page clearly identifies the 2025 Pulitzer Prize for Fiction winner as stated. Allow minor formatting or punctuation differences in the title or author name.",
    )

    # Record which award URL we used
    evaluator.add_custom_info(
        {"used_official_award_url": used_award_url or "None"},
        info_type="diagnostic",
        info_name="official_award_url_selected",
    )


async def verify_book_identity(
    evaluator: Evaluator,
    parent_node,
    info: PulitzerWinnerExtraction,
    combined_sources: List[str],
) -> None:
    """
    Build and verify the 'Book_Identity' section:
    - Title provided and supported by sources
    - Author provided and supported by sources
    """
    identity = evaluator.add_parallel(
        id="Book_Identity",
        desc="Correctly identify the winning book's title and author",
        parent=parent_node,
        critical=True,
    )

    # Title provided (critical)
    title_provided = evaluator.add_custom_node(
        result=bool(info.book_title and info.book_title.strip()),
        id="Book_Title_provided",
        desc="The book title is provided",
        parent=identity,
        critical=True,
    )

    # Title supported (critical)
    title_supported = evaluator.add_leaf(
        id="Book_Title_supported",
        desc="Provide the correct and complete title of the 2025 Pulitzer Prize for Fiction winner, supported by cited sources",
        parent=identity,
        critical=True,
    )

    title_claim = f"The book that won the 2025 Pulitzer Prize for Fiction has the title '{info.book_title}'."
    await evaluator.verify(
        claim=title_claim,
        node=title_supported,
        sources=combined_sources,
        additional_instruction="Verify the exact or equivalent title from the provided sources (especially the pulitzer.org page if included). Allow minor punctuation, casing, or subtitle formatting differences.",
    )

    # Author provided (critical)
    author_provided = evaluator.add_custom_node(
        result=bool(info.author_name and info.author_name.strip()),
        id="Author_Name_provided",
        desc="The author name is provided",
        parent=identity,
        critical=True,
    )

    # Author supported (critical)
    author_supported = evaluator.add_leaf(
        id="Author_Name_supported",
        desc="Provide the correct and complete author name of the 2025 Pulitzer Prize Fiction winner, supported by cited sources",
        parent=identity,
        critical=True,
    )

    if info.book_title:
        author_claim = f"The author of the 2025 Pulitzer Prize for Fiction winning book '{info.book_title}' is {info.author_name}."
    else:
        author_claim = f"The author of the 2025 Pulitzer Prize for Fiction winner is {info.author_name}."

    await evaluator.verify(
        claim=author_claim,
        node=author_supported,
        sources=combined_sources,
        additional_instruction="Confirm the author's full name in the provided sources. Allow minor variations (middle initials, accents, casing).",
    )


async def verify_publication_details(
    evaluator: Evaluator,
    parent_node,
    info: PulitzerWinnerExtraction,
    combined_sources: List[str],
) -> None:
    """
    Build and verify the 'Publication_Details' section:
    - Publisher provided and supported
    - Publication date provided, complete format, and supported
    """
    details = evaluator.add_parallel(
        id="Publication_Details",
        desc="Provide accurate publication information for cataloging purposes",
        parent=parent_node,
        critical=True,
    )

    # Publisher sub-node
    publisher_main = evaluator.add_parallel(
        id="Publisher_Name_main",
        desc="Publisher name checks",
        parent=details,
        critical=True,
    )

    publisher_provided = evaluator.add_custom_node(
        result=bool(info.publisher_name and info.publisher_name.strip()),
        id="Publisher_Name_provided",
        desc="Publisher name is provided",
        parent=publisher_main,
        critical=True,
    )

    publisher_supported = evaluator.add_leaf(
        id="Publisher_Name_supported",
        desc="Publisher name is accurately supported by cited sources",
        parent=publisher_main,
        critical=True,
    )

    if info.book_title:
        publisher_claim = f"The publisher of the 2025 Pulitzer Prize for Fiction winning book '{info.book_title}' is '{info.publisher_name}'."
    else:
        publisher_claim = f"The publisher of the 2025 Pulitzer Prize for Fiction winner is '{info.publisher_name}'."

    await evaluator.verify(
        claim=publisher_claim,
        node=publisher_supported,
        sources=combined_sources,
        additional_instruction="Prefer evidence from the publisher's official website or other authoritative bibliographic sources. Allow minor imprint naming variations.",
    )

    # Publication date sub-node
    pub_date_main = evaluator.add_parallel(
        id="Publication_Date_main",
        desc="Publication date checks",
        parent=details,
        critical=True,
    )

    pub_date_provided = evaluator.add_custom_node(
        result=bool(info.publication_date and info.publication_date.strip()),
        id="Publication_Date_provided",
        desc="Publication date is provided",
        parent=pub_date_main,
        critical=True,
    )

    pub_date_complete = evaluator.add_custom_node(
        result=is_complete_date(info.publication_date),
        id="Publication_Date_complete_format",
        desc="Publication date includes month, day, and year (complete date)",
        parent=pub_date_main,
        critical=True,
    )

    pub_date_supported = evaluator.add_leaf(
        id="Publication_Date_supported",
        desc="Publication date is accurately supported by cited sources",
        parent=pub_date_main,
        critical=True,
    )

    if info.book_title:
        date_claim = f"The publication date of the book '{info.book_title}' is '{info.publication_date}'."
    else:
        date_claim = f"The publication date of the winning book is '{info.publication_date}'."

    await evaluator.verify(
        claim=date_claim,
        node=pub_date_supported,
        sources=combined_sources,
        additional_instruction="Verify that at least one authoritative source (ideally the publisher or an official bibliographic record) provides this exact publication date. If multiple editions exist, accept the date clearly indicated for the book edition referenced by the source.",
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 2025 Pulitzer Prize for Fiction winner task.
    Returns the evaluator summary dict.
    """
    # Initialize evaluator (root is always non-critical in the framework)
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

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_winner_info(),
        template_class=PulitzerWinnerExtraction,
        extraction_name="pulitzer_2025_fiction_extraction",
    )

    # Build a critical main node (since root is non-critical by design)
    main_node = evaluator.add_parallel(
        id="2025_Pulitzer_Prize_Fiction_Winner_Information",
        desc="Provide complete information about the book that won the 2025 Pulitzer Prize for Fiction, including official award verification, book identity, and publication details",
        parent=root,
        critical=True,
    )

    # Prepare combined sources (official award URL + all other sources)
    used_award_url = pick_official_award_url(extracted)
    combined_sources = unique_urls(([used_award_url] if used_award_url else []) + (extracted.sources or []))

    evaluator.add_custom_info(
        {
            "used_award_url": used_award_url or "None",
            "total_sources_collected": len(extracted.sources or []),
            "combined_sources_used_for_verification": combined_sources,
        },
        info_type="diagnostic",
        info_name="source_overview",
    )

    # Official award section
    await verify_official_award_section(evaluator, main_node, extracted)

    # Book identity section
    await verify_book_identity(evaluator, main_node, extracted, combined_sources)

    # Publication details section
    await verify_publication_details(evaluator, main_node, extracted, combined_sources)

    # Return summary
    return evaluator.get_summary()