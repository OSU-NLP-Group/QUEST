import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "awards_2025_books"
TASK_DESCRIPTION = (
    "Identify four books from major 2025 English-language literary awards, providing complete verified information for each:\n\n"
    "1. The book that won the 2025 Booker Prize - include the title, author, UK publisher, page count, and the date in November 2025 "
    "when the winner was announced. Provide a reference URL from the official Booker Prize website.\n\n"
    "2. The book that won the 2025 National Book Award for Fiction - include the title, author, publisher (with imprint), the venue in "
    "New York City where the ceremony was held, and the ceremony date in November 2025. Provide a reference URL from the National Book "
    "Foundation website.\n\n"
    "3. The book that won the 2025 Pulitzer Prize for Fiction - include the title, author, publisher, publication date, and page count. "
    "Provide a reference URL from the official Pulitzer Prize website.\n\n"
    "4. One book from the 2025 Booker Prize shortlist that did not win - include the title, author, and the author's nationality or "
    "background. Verify that this book was shortlisted but did not win the prize. Provide a reference URL from the official Booker Prize website.\n\n"
    "For all books, reference URLs must be from official award websites or other authoritative sources that explicitly confirm the award status."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BookerWinner(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    uk_publisher: Optional[str] = None
    page_count: Optional[str] = None  # prefer strings like "368" or "368 pages"
    announcement_date: Optional[str] = None  # e.g., "November 10, 2025"
    official_url: Optional[str] = None  # official Booker Prize URL
    extra_urls: List[str] = Field(default_factory=list)  # other authoritative sources


class NBAFictionWinner(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher_imprint: Optional[str] = None  # e.g., "Grove Press / Grove Atlantic"
    ceremony_venue: Optional[str] = None  # e.g., "Cipriani Wall Street, NYC"
    ceremony_date: Optional[str] = None  # e.g., "November 20, 2025"
    official_url: Optional[str] = None  # National Book Foundation official URL
    extra_urls: List[str] = Field(default_factory=list)


class PulitzerFictionWinner(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None  # e.g., "Doubleday"
    publication_date: Optional[str] = None  # e.g., "March 19, 2024"
    page_count: Optional[str] = None  # e.g., "320" or "320 pages"
    official_url: Optional[str] = None  # official Pulitzer Prize URL
    extra_urls: List[str] = Field(default_factory=list)


class BookerShortlistNonWinner(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    author_nationality: Optional[str] = None  # nationality/background string
    official_url: Optional[str] = None  # official Booker Prize URL for the shortlisted book/list
    extra_urls: List[str] = Field(default_factory=list)


class AwardsExtraction(BaseModel):
    booker_winner: Optional[BookerWinner] = None
    nba_winner: Optional[NBAFictionWinner] = None
    pulitzer_winner: Optional[PulitzerFictionWinner] = None
    booker_shortlist_nonwinner: Optional[BookerShortlistNonWinner] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_awards() -> str:
    return """
    Extract structured information for four books from the answer, corresponding to the specified 2025 English-language literary awards.

    For each of the following, extract exactly what the answer states. If the answer does not provide the information, return null for that field.
    Also extract URLs explicitly mentioned in the answer. Do not invent URLs.

    1) 2025 Booker Prize winner:
       - title
       - author
       - uk_publisher (UK edition publisher/imprint)
       - page_count (as a string; e.g., "368" or "368 pages")
       - announcement_date (the November 2025 date when the winner was announced; e.g., "November 10, 2025")
       - official_url (must be a URL on the official Booker Prize site; e.g., thebookerprizes.com or bookerprize.com)
       - extra_urls (list of any additional authoritative URLs the answer cites for this book)

    2) 2025 National Book Award for Fiction winner:
       - title
       - author
       - publisher_imprint (including imprint, e.g., "Grove Press / Grove Atlantic")
       - ceremony_venue (e.g., "Cipriani Wall Street, New York City")
       - ceremony_date (the November 2025 ceremony date; e.g., "November 20, 2025")
       - official_url (must be a URL on nationalbook.org)
       - extra_urls (list of any additional authoritative URLs the answer cites for this book)

    3) 2025 Pulitzer Prize for Fiction winner:
       - title
       - author
       - publisher (e.g., "Doubleday")
       - publication_date (e.g., "March 19, 2024")
       - page_count (as a string; e.g., "320" or "320 pages")
       - official_url (must be a URL on pulitzer.org)
       - extra_urls (list of any additional authoritative URLs the answer cites for this book)

    4) 2025 Booker Prize shortlist (non-winner):
       - title
       - author
       - author_nationality (the author's nationality/background as stated)
       - official_url (must be a URL on the official Booker Prize site confirming shortlist status)
       - extra_urls (list of any additional authoritative URLs the answer cites for this book)

    Rules for URL extraction:
    - Extract only explicit URLs present in the answer. Accept plain URLs or URLs inside markdown links.
    - For official_url fields, ensure the domain matches the required official site (Booker: thebookerprizes.com or bookerprize.com; NBF: nationalbook.org; Pulitzer: pulitzer.org). If none is present, set official_url to null.
    - For extra_urls, include any additional URLs that the answer mentions for that book (publisher site, press releases, etc.).

    Return a JSON object with keys: booker_winner, nba_winner, pulitzer_winner, booker_shortlist_nonwinner.
    Each key should contain the corresponding object with the fields listed above.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_urls(primary: Optional[str], extras: Optional[List[str]]) -> List[str]:
    """Combine primary and extra URLs, filter empties, deduplicate."""
    urls: List[str] = []
    if primary and isinstance(primary, str) and primary.strip():
        urls.append(primary.strip())
    if extras:
        for u in extras:
            if isinstance(u, str) and u.strip():
                urls.append(u.strip())
    # Deduplicate while keeping order
    seen = set()
    unique_urls: List[str] = []
    for u in urls:
        if u not in seen:
            unique_urls.append(u)
            seen.add(u)
    return unique_urls


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_booker_winner(evaluator: Evaluator, root_node, info: Optional[BookerWinner]) -> None:
    """Build and verify the 2025 Booker Prize winner subtree."""
    parent = evaluator.add_parallel(
        id="item_1_booker_winner",
        desc="2025 Booker Prize winner (required metadata + official Booker citation).",
        parent=root_node,
        critical=False
    )

    title_val = info.title if info else ""
    author_val = info.author if info else ""
    uk_pub_val = info.uk_publisher if info else ""
    page_count_val = info.page_count if info else ""
    ann_date_val = info.announcement_date if info else ""
    official = info.official_url if info else None
    sources_all = _combine_urls(official, info.extra_urls if info else [])

    # Title
    node_title = evaluator.add_leaf(
        id="booker_winner_title",
        desc="Provide the title of the 2025 Booker Prize winner.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 2025 Booker Prize winner's title is '{title_val}'.",
        node=node_title,
        sources=sources_all,
        additional_instruction="Use the official Booker Prize page or authoritative sources to confirm the exact title. Allow minor casing variations."
    )

    # Author
    node_author = evaluator.add_leaf(
        id="booker_winner_author",
        desc="Provide the author of the 2025 Booker Prize winner.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The author of the 2025 Booker Prize winner is '{author_val}'.",
        node=node_author,
        sources=sources_all,
        additional_instruction="Confirm the named author from the official Booker Prize page or authoritative sources. Allow minor variants in naming (e.g., middle initials)."
    )

    # UK publisher (UK edition) verifiably Jonathan Cape
    node_publisher = evaluator.add_leaf(
        id="booker_winner_uk_publisher_jonathan_cape",
        desc="UK publisher (UK edition) is verifiably Jonathan Cape.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The UK edition publisher/imprint of the 2025 Booker Prize winner is '{uk_pub_val}'.",
        node=node_publisher,
        sources=sources_all,
        additional_instruction="Check authoritative sources (Booker page, publisher imprint pages). This should match Jonathan Cape for the UK edition; treat equivalent phrasing (imprint vs house) reasonably."
    )

    # Page count verifiably 368 pages
    node_pagecount = evaluator.add_leaf(
        id="booker_winner_page_count_368",
        desc="Page count is verifiably 368 pages.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page count of the book is {page_count_val} pages.",
        node=node_pagecount,
        sources=sources_all,
        additional_instruction="Confirm the numeric page count from authoritative sources (publisher listing, official pages). Accept '368' or '368 pages' as equivalent representations."
    )

    # Announcement date verifiably November 10, 2025
    node_date = evaluator.add_leaf(
        id="booker_winner_announcement_date_2025_11_10",
        desc="Winner announcement date is verifiably November 10, 2025.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 2025 Booker Prize winner was announced on {ann_date_val}.",
        node=node_date,
        sources=official or sources_all,
        additional_instruction="Verify the announcement date from the official Booker Prize site (e.g., winner announcement or press release). Allow date format variants such as '10 November 2025'."
    )

    # Official Booker URL verifies winner status
    node_status = evaluator.add_leaf(
        id="booker_winner_official_booker_url_verifies_winner_status",
        desc="Provide a reference URL from the official Booker Prize website that verifies the selected book is the 2025 Booker Prize winner.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"This official Booker Prize webpage confirms that '{title_val}' by {author_val} won the 2025 Booker Prize.",
        node=node_status,
        sources=official,
        additional_instruction="Pass only if the URL is on the official Booker Prize domain and explicitly confirms winner status."
    )


async def verify_nba_winner(evaluator: Evaluator, root_node, info: Optional[NBAFictionWinner]) -> None:
    """Build and verify the 2025 National Book Award for Fiction winner subtree."""
    parent = evaluator.add_parallel(
        id="item_2_nba_fiction_winner",
        desc="2025 National Book Award for Fiction winner (required metadata + National Book Foundation citation).",
        parent=root_node,
        critical=False
    )

    title_val = info.title if info else ""
    author_val = info.author if info else ""
    publisher_imprint_val = info.publisher_imprint if info else ""
    venue_val = info.ceremony_venue if info else ""
    ceremony_date_val = info.ceremony_date if info else ""
    official = info.official_url if info else None
    sources_all = _combine_urls(official, info.extra_urls if info else [])

    # Title
    node_title = evaluator.add_leaf(
        id="nba_winner_title",
        desc="Provide the title of the 2025 National Book Award for Fiction winner.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 2025 National Book Award for Fiction winner's title is '{title_val}'.",
        node=node_title,
        sources=sources_all,
        additional_instruction="Use the National Book Foundation official page to confirm the title. Allow minor casing variations."
    )

    # Author
    node_author = evaluator.add_leaf(
        id="nba_winner_author",
        desc="Provide the author of the 2025 National Book Award for Fiction winner.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The author of the 2025 National Book Award for Fiction winner is '{author_val}'.",
        node=node_author,
        sources=sources_all,
        additional_instruction="Confirm the named author from the National Book Foundation page. Allow minor variants in naming."
    )

    # Publisher (including imprint) verifiably Grove Press / Grove Atlantic
    node_publisher = evaluator.add_leaf(
        id="nba_winner_publisher_grove_press_grove_atlantic",
        desc="Publisher (including imprint) is verifiably Grove Press / Grove Atlantic.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The publisher/imprint for the winning book is '{publisher_imprint_val}'.",
        node=node_publisher,
        sources=sources_all,
        additional_instruction="Confirm the imprint/publisher from NBF or publisher sources. Treat 'Grove Press (an imprint of Grove Atlantic)' and 'Grove Press / Grove Atlantic' as equivalent."
    )

    # Ceremony venue verifiably Cipriani Wall Street, NYC
    node_venue = evaluator.add_leaf(
        id="nba_ceremony_venue_cipriani_wall_street_nyc",
        desc="Ceremony venue is verifiably Cipriani Wall Street in New York City.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 2025 National Book Awards ceremony was held at {venue_val}.",
        node=node_venue,
        sources=official or sources_all,
        additional_instruction="Confirm the venue from National Book Foundation announcements/event page. Accept minor wording differences like 'Cipriani Wall Street, New York City'."
    )

    # Ceremony date verifiably November 20, 2025
    node_date = evaluator.add_leaf(
        id="nba_ceremony_date_2025_11_20",
        desc="Ceremony date is verifiably November 20, 2025.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 2025 National Book Awards ceremony date was {ceremony_date_val}.",
        node=node_date,
        sources=official or sources_all,
        additional_instruction="Verify the ceremony date from the National Book Foundation site. Accept date format variants like 'November 20, 2025' / '20 November 2025'."
    )

    # Official NBF URL verifies winner status
    node_status = evaluator.add_leaf(
        id="nba_official_nbf_url_verifies_winner_status",
        desc="Provide a reference URL from the National Book Foundation website that verifies the selected book is the 2025 National Book Award for Fiction winner.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"This National Book Foundation webpage confirms that '{title_val}' by {author_val} won the 2025 National Book Award for Fiction.",
        node=node_status,
        sources=official,
        additional_instruction="Pass only if the URL is on nationalbook.org and explicitly confirms the winner status."
    )


async def verify_pulitzer_winner(evaluator: Evaluator, root_node, info: Optional[PulitzerFictionWinner]) -> None:
    """Build and verify the 2025 Pulitzer Prize for Fiction winner subtree."""
    parent = evaluator.add_parallel(
        id="item_3_pulitzer_fiction_winner",
        desc="2025 Pulitzer Prize for Fiction winner (required metadata + official Pulitzer citation).",
        parent=root_node,
        critical=False
    )

    title_val = info.title if info else ""
    author_val = info.author if info else ""
    publisher_val = info.publisher if info else ""
    pub_date_val = info.publication_date if info else ""
    page_count_val = info.page_count if info else ""
    official = info.official_url if info else None
    sources_all = _combine_urls(official, info.extra_urls if info else [])

    # Title
    node_title = evaluator.add_leaf(
        id="pulitzer_winner_title",
        desc="Provide the title of the 2025 Pulitzer Prize for Fiction winner.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 2025 Pulitzer Prize for Fiction winner's title is '{title_val}'.",
        node=node_title,
        sources=sources_all,
        additional_instruction="Use the official Pulitzer Prize site to confirm the title. Allow minor casing variations."
    )

    # Author
    node_author = evaluator.add_leaf(
        id="pulitzer_winner_author",
        desc="Provide the author of the 2025 Pulitzer Prize for Fiction winner.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The author of the 2025 Pulitzer Prize for Fiction winner is '{author_val}'.",
        node=node_author,
        sources=sources_all,
        additional_instruction="Confirm the author name from the official Pulitzer Prize site or authoritative sources."
    )

    # Publisher verifiably Doubleday
    node_publisher = evaluator.add_leaf(
        id="pulitzer_winner_publisher_doubleday",
        desc="Publisher is verifiably Doubleday.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The publisher of the winning book is '{publisher_val}'.",
        node=node_publisher,
        sources=sources_all,
        additional_instruction="Confirm the publisher from authoritative sources (Pulitzer page may reference the book; publisher listing is acceptable). This should match 'Doubleday'."
    )

    # Publication date verifiably March 19, 2024
    node_pubdate = evaluator.add_leaf(
        id="pulitzer_winner_publication_date_2024_03_19",
        desc="Publication date is verifiably March 19, 2024.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The publication date of the book is {pub_date_val}.",
        node=node_pubdate,
        sources=sources_all,
        additional_instruction="Confirm publication date from authoritative sources (publisher listing, bibliographic records). Accept format variants like 'March 19, 2024' / '19 March 2024'."
    )

    # Page count verifiably 320 pages
    node_pagecount = evaluator.add_leaf(
        id="pulitzer_winner_page_count_320",
        desc="Page count is verifiably 320 pages.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page count of the book is {page_count_val} pages.",
        node=node_pagecount,
        sources=sources_all,
        additional_instruction="Confirm the numeric page count from authoritative sources (publisher site, bibliographic records). Accept '320' or '320 pages' as equivalent representations."
    )

    # Official Pulitzer URL verifies winner status
    node_status = evaluator.add_leaf(
        id="pulitzer_official_url_verifies_winner_status",
        desc="Provide a reference URL from the official Pulitzer Prize website that verifies the selected book is the 2025 Pulitzer Prize for Fiction winner.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"This official Pulitzer Prize webpage confirms that '{title_val}' by {author_val} won the 2025 Pulitzer Prize for Fiction.",
        node=node_status,
        sources=official,
        additional_instruction="Pass only if the URL is on pulitzer.org and explicitly confirms winner status."
    )


async def verify_booker_shortlist_nonwinner(evaluator: Evaluator, root_node, info: Optional[BookerShortlistNonWinner]) -> None:
    """Build and verify one 2025 Booker Prize shortlisted book that did not win."""
    parent = evaluator.add_parallel(
        id="item_4_booker_shortlist_nonwinner",
        desc="One 2025 Booker Prize shortlisted book that did not win (required metadata + official Booker citation).",
        parent=root_node,
        critical=False
    )

    title_val = info.title if info else ""
    author_val = info.author if info else ""
    nationality_val = info.author_nationality if info else ""
    official = info.official_url if info else None
    sources_all = _combine_urls(official, info.extra_urls if info else [])

    # Title
    node_title = evaluator.add_leaf(
        id="shortlist_book_title",
        desc="Provide the title of a 2025 Booker Prize shortlisted book that is not the winner.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The shortlisted (non-winning) book's title is '{title_val}'.",
        node=node_title,
        sources=sources_all,
        additional_instruction="Confirm the title from the official Booker Prize site shortlist page or the book's profile."
    )

    # Author
    node_author = evaluator.add_leaf(
        id="shortlist_book_author",
        desc="Provide the author of the shortlisted non-winning book.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The author of the shortlisted (non-winning) book is '{author_val}'.",
        node=node_author,
        sources=sources_all,
        additional_instruction="Confirm the author's name from the official Booker Prize site shortlist page or the book's profile."
    )

    # Author nationality/background
    node_nat = evaluator.add_leaf(
        id="author_nationality_or_background",
        desc="Provide the author's nationality or background, verifiably supported by a cited source.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The author's nationality/background is '{nationality_val}'.",
        node=node_nat,
        sources=sources_all,
        additional_instruction="Verify nationality/background from authoritative sources (Booker author page, publisher bio, reputable biographies). Allow minor phrasing variations."
    )

    # Official Booker URL verifies shortlist and non-winner status
    node_status = evaluator.add_leaf(
        id="shortlist_official_booker_url_verifies_shortlist_and_nonwinner_status",
        desc="Provide a reference URL from the official Booker Prize website that verifies the selected book was shortlisted for the 2025 Booker Prize and did not win.",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"This official Booker Prize webpage confirms that '{title_val}' was shortlisted for the 2025 Booker Prize and did not win.",
        node=node_status,
        sources=official or sources_all,
        additional_instruction="Pass only if the page on the official Booker Prize site explicitly indicates shortlist status for 2025 and the book is not identified as the winner. Using a shortlist page or the individual shortlisted book page is acceptable."
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 2025 awards books task.
    """
    # Initialize evaluator with root parallel aggregation
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

    # Extract structured information once for all four items
    extraction = await evaluator.extract(
        prompt=prompt_extract_awards(),
        template_class=AwardsExtraction,
        extraction_name="awards_2025_extraction"
    )

    # Add basic ground truth guidance (domains policy) to summary
    evaluator.add_ground_truth({
        "required_official_domains": {
            "booker": ["thebookerprizes.com", "bookerprize.com"],
            "national_book_foundation": ["nationalbook.org"],
            "pulitzer": ["pulitzer.org"]
        },
        "policy": "Award status must be verified on official sites; other metadata may be supported by additional authoritative sources."
    }, gt_type="verification_policy")

    # Build verification subtrees
    await verify_booker_winner(evaluator, root, extraction.booker_winner)
    await verify_nba_winner(evaluator, root, extraction.nba_winner)
    await verify_pulitzer_winner(evaluator, root, extraction.pulitzer_winner)
    await verify_booker_shortlist_nonwinner(evaluator, root, extraction.booker_shortlist_nonwinner)

    # Return final structured summary
    return evaluator.get_summary()