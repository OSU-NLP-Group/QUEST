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
TASK_ID = "literary_fiction_2024_2025"
TASK_DESCRIPTION = (
    "Identify three major award-winning literary fiction books from the 2024-2025 period by researching official award announcements "
    "and authoritative literary sources. Specifically, you must identify:\n\n"
    "1. The 2024 National Book Award for Fiction winner: Provide the book title, author, publisher name (verify it is a major US publisher), "
    "publication date in 2024, page count, and a reference URL confirming the award win; and verify the ceremony was held on Nov 20, 2024 at "
    "Cipriani Wall Street in New York.\n"
    "2. The 2024 Booker Prize winner: Provide the book title, author, publisher name, page count, the novel's genre/type, and a reference URL "
    "confirming the award win; and verify the ceremony was held on Nov 12, 2024 at Old Billingsgate in London.\n"
    "3. The 2024 Pulitzer Prize for Fiction winner: Provide the book title, author, publisher name (as mentioned in parentheses on the official page), "
    "and a reference URL confirming the award win; and verify the award was announced on May 6, 2024.\n"
    "Use authoritative sources: National Book Foundation, The Booker Prizes official website, The Pulitzer Prizes official website, major news publications, "
    "or established literary databases."
)

# Ground-truth ceremony dates/venues (for logging/reference only)
GROUND_TRUTH_INFO = {
    "NBA_2024": {
        "ceremony_date": "November 20, 2024",
        "ceremony_venue": "Cipriani Wall Street, New York"
    },
    "Booker_2024": {
        "ceremony_date": "November 12, 2024",
        "ceremony_venue": "Old Billingsgate, London"
    },
    "Pulitzer_2024": {
        "award_announcement_date": "May 6, 2024"
    }
}


# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class AwardEntryNBA(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    publication_date: Optional[str] = None
    page_count: Optional[str] = None
    ceremony_date: Optional[str] = None
    ceremony_venue: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    additional_urls: List[str] = Field(default_factory=list)


class AwardEntryBooker(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    page_count: Optional[str] = None
    genre: Optional[str] = None
    ceremony_date: Optional[str] = None
    ceremony_venue: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    additional_urls: List[str] = Field(default_factory=list)


class AwardEntryPulitzer(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    award_announcement_date: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    additional_urls: List[str] = Field(default_factory=list)


class MasterExtraction(BaseModel):
    national_book_award_2024: Optional[AwardEntryNBA] = None
    booker_prize_2024: Optional[AwardEntryBooker] = None
    pulitzer_prize_fiction_2024: Optional[AwardEntryPulitzer] = None


# --------------------------------------------------------------------------- #
# Extraction Prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_master() -> str:
    return """
    Extract structured information for THREE distinct award-winning literary fiction books from the 2024-2025 period, as presented in the answer.
    You must extract fields for the following awards:

    A) 2024 National Book Award for Fiction (NBA):
       - title
       - author
       - publisher
       - publication_date (must be in 2024)
       - page_count
       - ceremony_date (e.g., "November 20, 2024")
       - ceremony_venue (e.g., "Cipriani Wall Street, New York")
       - reference_urls: URLs that explicitly confirm the award win (e.g., nationalbook.org, major news)
       - additional_urls: any other URLs that provide bibliographic details (publisher, page count, publication date)

    B) 2024 Booker Prize (Booker):
       - title
       - author
       - publisher
       - page_count
       - genre (e.g., literary fiction, science fiction, historical novel, etc.)
       - ceremony_date (e.g., "November 12, 2024")
       - ceremony_venue (e.g., "Old Billingsgate, London")
       - reference_urls: URLs that explicitly confirm the award win (e.g., thebookerprizes.com, major news)
       - additional_urls: any other URLs providing bibliographic details

    C) 2024 Pulitzer Prize for Fiction:
       - title
       - author
       - publisher (as mentioned in parentheses in official Pulitzer documentation if available)
       - award_announcement_date (e.g., "May 6, 2024")
       - reference_urls: URLs that explicitly confirm the award win (e.g., pulitzer.org)
       - additional_urls: any other URLs providing bibliographic details

    IMPORTANT:
    - Extract only information explicitly present in the answer. Do not invent data.
    - Return null for any field not present in the answer.
    - For reference_urls/additional_urls, extract actual URLs visible in the answer text; support plain or markdown links.
    - Do not deduplicate; keep URLs as they appear; we will handle duplicates later.

    Return a JSON object with keys:
    - national_book_award_2024: object with fields above for NBA
    - booker_prize_2024: object with fields above for Booker
    - pulitzer_prize_fiction_2024: object with fields above for Pulitzer
    """


# --------------------------------------------------------------------------- #
# Helper Utilities                                                            #
# --------------------------------------------------------------------------- #
def merge_urls(*url_lists: List[str]) -> List[str]:
    """Merge multiple URL lists and deduplicate, preserving order."""
    seen = set()
    merged: List[str] = []
    for urls in url_lists:
        for u in urls or []:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls)


def non_empty(s: Optional[str]) -> bool:
    return bool(s) and bool(str(s).strip())


# --------------------------------------------------------------------------- #
# Verification Builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_national_book_award_2024(
    evaluator: Evaluator,
    parent_node,
    nba: Optional[AwardEntryNBA],
) -> None:
    # Create award-level parallel node
    nba_node = evaluator.add_parallel(
        id="National_Book_Award_Fiction_Winner_2024",
        desc="Identify the book that won the National Book Award for Fiction in 2024, verify all required details",
        parent=parent_node,
        critical=False
    )

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=nba is not None,
        id="nba_entry_exists",
        desc="NBA entry exists in the answer",
        parent=nba_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=non_empty(getattr(nba, "title", None)),
        id="nba_title_exists",
        desc="NBA: Title is provided",
        parent=nba_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=non_empty(getattr(nba, "author", None)),
        id="nba_author_exists",
        desc="NBA: Author is provided",
        parent=nba_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_url(getattr(nba, "reference_urls", None)),
        id="nba_ref_urls_exist",
        desc="NBA: Reference URL(s) provided",
        parent=nba_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=non_empty(getattr(nba, "publisher", None)),
        id="nba_publisher_exists",
        desc="NBA: Publisher is provided",
        parent=nba_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=non_empty(getattr(nba, "publication_date", None)),
        id="nba_pubdate_exists",
        desc="NBA: Publication date is provided",
        parent=nba_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=non_empty(getattr(nba, "page_count", None)),
        id="nba_page_count_exists",
        desc="NBA: Page count is provided",
        parent=nba_node,
        critical=True
    )

    # Prepare URLs
    ref_urls = (nba.reference_urls if nba else [])  # type: ignore
    all_urls = merge_urls(ref_urls, (nba.additional_urls if nba else []))  # type: ignore

    # Title verification
    title_leaf = evaluator.add_leaf(
        id="Title",
        desc="Provide the book title",
        parent=nba_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 2024 National Book Award for Fiction winner is titled '{nba.title}'.",
        node=title_leaf,
        sources=ref_urls,
        additional_instruction="Verify that the authoritative reference page explicitly names the winning book title for the 2024 National Book Award for Fiction. Allow minor formatting variants."
    )

    # Author verification
    author_leaf = evaluator.add_leaf(
        id="Author",
        desc="Provide the author's name",
        parent=nba_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The author of the 2024 National Book Award for Fiction winner is {nba.author}.",
        node=author_leaf,
        sources=ref_urls,
        additional_instruction="Verify that the authoritative reference page explicitly lists the author's name for the winning title. Allow minor formatting or casing differences."
    )

    # Award Information verification
    award_info_leaf = evaluator.add_leaf(
        id="Award_Information",
        desc="Book won the National Book Award for Fiction in 2024",
        parent=nba_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The book '{nba.title}' by {nba.author} won the National Book Award for Fiction in 2024.",
        node=award_info_leaf,
        sources=ref_urls,
        additional_instruction="Confirm the page explicitly states the book is the 2024 National Book Award for Fiction winner."
    )

    # Ceremony Details verification (date + venue as a single check per rubric)
    ceremony_leaf = evaluator.add_leaf(
        id="Ceremony_Details",
        desc="Award ceremony was held on November 20, 2024, at Cipriani Wall Street in New York",
        parent=nba_node,
        critical=True
    )
    await evaluator.verify(
        claim="The 2024 National Book Awards ceremony for the Fiction category was held on November 20, 2024 at Cipriani Wall Street in New York.",
        node=ceremony_leaf,
        sources=ref_urls,
        additional_instruction="Check official National Book Foundation pages or authoritative coverage to validate BOTH the date (Nov 20, 2024) and venue (Cipriani Wall Street, New York)."
    )

    # Publisher Information verification (name + major US publisher confirmation)
    publisher_leaf = evaluator.add_leaf(
        id="Publisher_Information",
        desc="Provide the publisher name and confirm it is a major US publisher",
        parent=nba_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The publisher of the winning book is '{nba.publisher}', and it is a major US publisher.",
        node=publisher_leaf,
        sources=all_urls,
        additional_instruction="Confirm the publisher name from the provided sources. Consider a 'major US publisher' to include Big Five houses (Penguin Random House, HarperCollins, Simon & Schuster, Macmillan, Hachette) and their primary imprints. Use only URL content."
    )

    # Publication Date verification (in 2024)
    pubdate_leaf = evaluator.add_leaf(
        id="Publication_Date",
        desc="Provide the publication date in 2024",
        parent=nba_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The publication date of the winning book is in 2024 (specifically '{nba.publication_date}').",
        node=pubdate_leaf,
        sources=all_urls,
        additional_instruction="Verify from publisher/book pages or credible databases that the publication date falls in the calendar year 2024. If only year is shown and it's 2024, accept."
    )

    # Page Count verification
    pagecount_leaf = evaluator.add_leaf(
        id="Page_Count",
        desc="Provide the page count of the book",
        parent=nba_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The book has {nba.page_count} pages.",
        node=pagecount_leaf,
        sources=all_urls,
        additional_instruction="Verify the page count from publisher or credible bibliographic sources. Allow reasonable variations across editions."
    )

    # Reference URL verification (supportiveness)
    refurl_leaf = evaluator.add_leaf(
        id="Reference_URL",
        desc="Provide a reference URL that verifies the award win and book details",
        parent=nba_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided reference URL(s) explicitly confirm the award win and include key bibliographic details (title and author).",
        node=refurl_leaf,
        sources=ref_urls,
        additional_instruction="Ensure the reference URL(s) are authoritative and explicitly confirm the 2024 NBA Fiction winner and the book’s title/author."
    )


async def verify_booker_prize_2024(
    evaluator: Evaluator,
    parent_node,
    booker: Optional[AwardEntryBooker],
) -> None:
    # Create award-level parallel node
    booker_node = evaluator.add_parallel(
        id="Booker_Prize_Winner_2024",
        desc="Identify the book that won the Booker Prize in 2024, verify all required details",
        parent=parent_node,
        critical=False
    )

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=booker is not None,
        id="booker_entry_exists",
        desc="Booker entry exists in the answer",
        parent=booker_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=non_empty(getattr(booker, "title", None)),
        id="booker_title_exists",
        desc="Booker: Title is provided",
        parent=booker_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=non_empty(getattr(booker, "author", None)),
        id="booker_author_exists",
        desc="Booker: Author is provided",
        parent=booker_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_url(getattr(booker, "reference_urls", None)),
        id="booker_ref_urls_exist",
        desc="Booker: Reference URL(s) provided",
        parent=booker_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=non_empty(getattr(booker, "publisher", None)),
        id="booker_publisher_exists",
        desc="Booker: Publisher is provided",
        parent=booker_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=non_empty(getattr(booker, "page_count", None)),
        id="booker_page_count_exists",
        desc="Booker: Page count is provided",
        parent=booker_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=non_empty(getattr(booker, "genre", None)),
        id="booker_genre_exists",
        desc="Booker: Genre/type is provided",
        parent=booker_node,
        critical=True
    )

    # Prepare URLs
    ref_urls = (booker.reference_urls if booker else [])  # type: ignore
    all_urls = merge_urls(ref_urls, (booker.additional_urls if booker else []))  # type: ignore

    # Title verification
    title_leaf = evaluator.add_leaf(
        id="Title",
        desc="Provide the book title",
        parent=booker_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 2024 Booker Prize winner is titled '{booker.title}'.",
        node=title_leaf,
        sources=ref_urls,
        additional_instruction="Verify the official Booker page or authoritative coverage names the winning title for 2024."
    )

    # Author verification
    author_leaf = evaluator.add_leaf(
        id="Author",
        desc="Provide the author's name",
        parent=booker_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The author of the 2024 Booker Prize winner is {booker.author}.",
        node=author_leaf,
        sources=ref_urls,
        additional_instruction="Verify the official Booker page lists the author of the winning title."
    )

    # Award Information verification
    award_info_leaf = evaluator.add_leaf(
        id="Award_Information",
        desc="Book won the Booker Prize in 2024",
        parent=booker_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The book '{booker.title}' by {booker.author} won the Booker Prize in 2024.",
        node=award_info_leaf,
        sources=ref_urls,
        additional_instruction="Confirm the page explicitly states the book is the 2024 Booker Prize winner."
    )

    # Ceremony Details verification (date + venue per rubric)
    ceremony_leaf = evaluator.add_leaf(
        id="Ceremony_Details",
        desc="Award ceremony was held on November 12, 2024, at Old Billingsgate in London",
        parent=booker_node,
        critical=True
    )
    await evaluator.verify(
        claim="The 2024 Booker Prize ceremony was held on November 12, 2024 at Old Billingsgate in London.",
        node=ceremony_leaf,
        sources=ref_urls,
        additional_instruction="Check the official Booker site or authoritative coverage to validate BOTH the date (Nov 12, 2024) and venue (Old Billingsgate, London)."
    )

    # Publisher Information verification
    publisher_leaf = evaluator.add_leaf(
        id="Publisher_Information",
        desc="Provide the publisher name",
        parent=booker_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The publisher of the winning book is '{booker.publisher}'.",
        node=publisher_leaf,
        sources=all_urls,
        additional_instruction="Confirm the publisher name from the official page or credible bibliographic sources."
    )

    # Page Count verification
    pagecount_leaf = evaluator.add_leaf(
        id="Page_Count",
        desc="Provide the page count of the book",
        parent=booker_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The book has {booker.page_count} pages.",
        node=pagecount_leaf,
        sources=all_urls,
        additional_instruction="Confirm the page count from the official publisher page or credible bibliographic sources. Allow reasonable variation across editions."
    )

    # Genre/type verification
    genre_leaf = evaluator.add_leaf(
        id="Novel_Genre_Type",
        desc="Indicate the genre or type of novel (e.g., literary fiction, science fiction, etc.)",
        parent=booker_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The genre/type of the winning book is '{booker.genre}'.",
        node=genre_leaf,
        sources=all_urls,
        additional_instruction="Confirm the genre/type from the official page or credible coverage; accept 'novel' or recognized literary fiction descriptors."
    )

    # Reference URL verification (supportiveness)
    refurl_leaf = evaluator.add_leaf(
        id="Reference_URL",
        desc="Provide a reference URL that verifies the award win and book details",
        parent=booker_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided reference URL(s) explicitly confirm the award win and include key bibliographic details (title and author).",
        node=refurl_leaf,
        sources=ref_urls,
        additional_instruction="Ensure the reference URL(s) are authoritative and explicitly confirm the 2024 Booker Prize winner and the book’s title/author."
    )


async def verify_pulitzer_prize_fiction_2024(
    evaluator: Evaluator,
    parent_node,
    pulitzer: Optional[AwardEntryPulitzer],
) -> None:
    # Create award-level parallel node
    pulitzer_node = evaluator.add_parallel(
        id="Pulitzer_Prize_Fiction_Winner_2024",
        desc="Identify the book that won the Pulitzer Prize for Fiction in 2024, verify all required details",
        parent=parent_node,
        critical=False
    )

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=pulitzer is not None,
        id="pulitzer_entry_exists",
        desc="Pulitzer entry exists in the answer",
        parent=pulitzer_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=non_empty(getattr(pulitzer, "title", None)),
        id="pulitzer_title_exists",
        desc="Pulitzer: Title is provided",
        parent=pulitzer_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=non_empty(getattr(pulitzer, "author", None)),
        id="pulitzer_author_exists",
        desc="Pulitzer: Author is provided",
        parent=pulitzer_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_url(getattr(pulitzer, "reference_urls", None)),
        id="pulitzer_ref_urls_exist",
        desc="Pulitzer: Reference URL(s) provided",
        parent=pulitzer_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=non_empty(getattr(pulitzer, "publisher", None)),
        id="pulitzer_publisher_exists",
        desc="Pulitzer: Publisher is provided",
        parent=pulitzer_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=non_empty(getattr(pulitzer, "award_announcement_date", None)),
        id="pulitzer_award_date_exists",
        desc="Pulitzer: Award announcement date is provided",
        parent=pulitzer_node,
        critical=True
    )

    # Prepare URLs
    ref_urls = (pulitzer.reference_urls if pulitzer else [])  # type: ignore
    all_urls = merge_urls(ref_urls, (pulitzer.additional_urls if pulitzer else []))  # type: ignore

    # Title verification
    title_leaf = evaluator.add_leaf(
        id="Title",
        desc="Provide the book title",
        parent=pulitzer_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 2024 Pulitzer Prize for Fiction winner is titled '{pulitzer.title}'.",
        node=title_leaf,
        sources=ref_urls,
        additional_instruction="Verify the official Pulitzer page or authoritative coverage names the winning title for 2024."
    )

    # Author verification
    author_leaf = evaluator.add_leaf(
        id="Author",
        desc="Provide the author's name",
        parent=pulitzer_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The author of the 2024 Pulitzer Prize for Fiction winner is {pulitzer.author}.",
        node=author_leaf,
        sources=ref_urls,
        additional_instruction="Verify the official Pulitzer page lists the author of the winning title."
    )

    # Award Information verification
    award_info_leaf = evaluator.add_leaf(
        id="Award_Information",
        desc="Book won the Pulitzer Prize for Fiction in 2024",
        parent=pulitzer_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The book '{pulitzer.title}' by {pulitzer.author} won the Pulitzer Prize for Fiction in 2024.",
        node=award_info_leaf,
        sources=ref_urls,
        additional_instruction="Confirm the page explicitly states the book is the 2024 Pulitzer Prize for Fiction winner."
    )

    # Publisher Information verification (parentheses requirement)
    publisher_leaf = evaluator.add_leaf(
        id="Publisher_Information",
        desc="Provide the publisher name and verify it is mentioned in parentheses in official Pulitzer Prize documentation",
        parent=pulitzer_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"On the official Pulitzer Prizes page for the 2024 Fiction winner, the publisher '{pulitzer.publisher}' appears in parentheses next to the title.",
        node=publisher_leaf,
        sources=ref_urls,
        additional_instruction="Inspect the official Pulitzer winner page text: the publisher should appear in parentheses adjacent to the title. Verify publisher string matches."
    )

    # Award Announcement Date verification
    award_date_leaf = evaluator.add_leaf(
        id="Award_Announcement_Date",
        desc="The Pulitzer Prize for Fiction 2024 was announced on May 6, 2024",
        parent=pulitzer_node,
        critical=True
    )
    await evaluator.verify(
        claim="The 2024 Pulitzer Prize for Fiction was announced on May 6, 2024.",
        node=award_date_leaf,
        sources=ref_urls,
        additional_instruction="Check the official Pulitzer site or authoritative coverage to confirm the announcement date as May 6, 2024."
    )

    # Reference URL verification (supportiveness)
    refurl_leaf = evaluator.add_leaf(
        id="Reference_URL",
        desc="Provide a reference URL that verifies the award win and book details",
        parent=pulitzer_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided reference URL(s) explicitly confirm the award win and include key bibliographic details (title and author).",
        node=refurl_leaf,
        sources=ref_urls,
        additional_instruction="Ensure the reference URL(s) explicitly confirm the 2024 Pulitzer Prize for Fiction winner and the book’s title/author."
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
    Evaluate an answer for the literary fiction award winners task (2024-2025).
    """
    # Initialize evaluator with parallel root according to rubric
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

    # Add ground truth info (ceremony dates/venues for logging)
    evaluator.add_ground_truth(GROUND_TRUTH_INFO, gt_type="ceremony_ground_truth")

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_master(),
        template_class=MasterExtraction,
        extraction_name="award_winners_2024",
    )

    # Build top-level node to match rubric naming
    top_node = evaluator.add_parallel(
        id="Literary_Fiction_Books_2024_2025",
        desc="Identify three award-winning literary fiction books from 2024-2025 period, each winning a different major literary award, published by major publishers, with complete bibliographic and award ceremony details",
        parent=root,
        critical=False
    )

    # Verify each award branch
    await verify_national_book_award_2024(evaluator, top_node, extracted.national_book_award_2024)
    await verify_booker_prize_2024(evaluator, top_node, extracted.booker_prize_2024)
    await verify_pulitzer_prize_fiction_2024(evaluator, top_node, extracted.pulitzer_prize_fiction_2024)

    # Return summary
    return evaluator.get_summary()