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
TASK_ID = "major_awards_2024_books"
TASK_DESCRIPTION = (
    "Identify 4 fiction books that won major literary awards in 2024. The major awards to consider are: "
    "the National Book Award for Fiction, the Booker Prize, the Pulitzer Prize for Fiction, or the "
    "PEN/Faulkner Award for Fiction. For each book, provide: (1) the specific award won, "
    "(2) the author's full name, (3) the publisher's name, (4) the exact publication date (month, day, year), "
    "(5) whether it was selected by Reese's Book Club or Oprah's Book Club in 2024 or 2025 (specify which and when if applicable; "
    'otherwise state "Not selected by Reese\'s Book Club or Oprah\'s Book Club"), '
    "(6) whether it has been adapted or is being adapted for screen (film/TV), including platform/status if applicable "
    '(otherwise state "No adaptation information found"), '
    "(7) a URL from the official award foundation website or reputable news source confirming the award win, and "
    "(8) a URL from the publisher website/book retailer/other reputable source confirming the publication details. "
    "All information must be verifiable through the provided URLs."
)

ALLOWED_AWARDS = [
    "National Book Award for Fiction",
    "Booker Prize",
    "Pulitzer Prize for Fiction",
    "PEN/Faulkner Award for Fiction",
]

TARGET_YEAR = 2024

# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class BookEntry(BaseModel):
    title: Optional[str] = None
    award: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    publication_date: Optional[str] = None  # Keep as string to be robust to formatting
    book_club_selection: Optional[str] = None  # e.g., "Reese's Book Club — March 2024", "Oprah's Book Club — Jan 2025", or the required "Not selected..." string
    adaptation_status: Optional[str] = None  # e.g., "In development as a Netflix series", "No adaptation information found"
    award_urls: List[str] = Field(default_factory=list)  # Official foundation or reputable news for the award
    publication_urls: List[str] = Field(default_factory=list)  # Publisher/retailer/reputable catalog confirming pub details
    club_urls: List[str] = Field(default_factory=list)  # Optional: sources for book club selection
    adaptation_urls: List[str] = Field(default_factory=list)  # Optional: sources for adaptation claims


class BooksExtraction(BaseModel):
    books: List[BookEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return """
    Extract up to four books from the answer that won a qualifying major literary award in 2024.
    For each book, return the following fields:

    - title: Exact book title as stated.
    - award: The specific award won (must be one of: National Book Award for Fiction; Booker Prize; Pulitzer Prize for Fiction; PEN/Faulkner Award for Fiction).
    - author: The author's full name as stated.
    - publisher: The publisher's name (imprint is fine, e.g., "Riverhead Books" under PRH).
    - publication_date: Exact publication date in Month Day, Year format as provided by the answer (e.g., "May 7, 2024"). If the answer provides a different but precise format, keep it exactly.
    - book_club_selection: If selected by Reese's Book Club or Oprah's Book Club in 2024 or 2025, specify "Reese's Book Club — Month Year" or "Oprah's Book Club — Month Year". Otherwise return exactly: "Not selected by Reese's Book Club or Oprah's Book Club".
    - adaptation_status: If adapted or being adapted for screen, provide a concise description including platform or status (e.g., "Film in development at A24", "Netflix limited series announced"). Otherwise return exactly: "No adaptation information found".
    - award_urls: Array of URLs that explicitly confirm the award win (prefer official foundation websites: nationalbook.org, pulitzer.org, thebookerprizes.com, penfaulkner.org; otherwise reputable news sources). Include all such URLs cited in the answer for this book.
    - publication_urls: Array of URLs that confirm publisher and publication date (prefer publisher sites; otherwise reputable retailer/catalog pages like Penguin Random House, HarperCollins, Macmillan, Hachette, Simon & Schuster, Barnes & Noble, Bookshop.org, Amazon, or library catalogs).
    - club_urls: (Optional) Array of URLs that confirm the book club selection claim (official club sites or reputable news). If none provided, return an empty array.
    - adaptation_urls: (Optional) Array of URLs that confirm adaptation info (e.g., studio/streamer press pages or reputable trades like Variety/THR/Deadline). If none provided, return an empty array.

    IMPORTANT:
    - Only extract information explicitly present in the answer text.
    - Each URL must be explicitly cited in the answer; do not invent URLs.
    - If a field is missing in the answer, set it to null (or empty array for URL lists).
    - Return a JSON object with a single field "books" which is an array of the extracted book objects in the exact order they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _merge_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if isinstance(u, str):
                u = u.strip()
                if u and u not in seen:
                    seen.add(u)
                    merged.append(u)
    return merged


def _safe(s: Optional[str]) -> str:
    return s or ""


# --------------------------------------------------------------------------- #
# Verification logic per book                                                 #
# --------------------------------------------------------------------------- #
async def verify_book(evaluator: Evaluator, parent_node, book: BookEntry, idx_zero_based: int) -> None:
    idx = idx_zero_based + 1
    title = _safe(book.title) or f"Book #{idx}"
    author = _safe(book.author)
    award = _safe(book.award)
    publisher = _safe(book.publisher)
    pub_date = _safe(book.publication_date)
    club_sel = _safe(book.book_club_selection)
    adapt = _safe(book.adaptation_status)

    # Create the main book node (parallel per rubric)
    book_node = evaluator.add_parallel(
        id=f"book_{idx}",
        desc=f"Book #{idx}: award-winning fiction book with required information",
        parent=parent_node,
        critical=False
    )

    # -------------------- Award Section --------------------
    award_section = evaluator.add_parallel(
        id=f"book_{idx}_award_section",
        desc=f"Book #{idx} award validation",
        parent=book_node,
        critical=False
    )

    # 1) Check award name is among allowed (critical leaf as per rubric)
    award_name_leaf = evaluator.add_leaf(
        id=f"book_{idx}_award_name",
        desc="The specified award is one of the allowed major awards",
        parent=award_section,
        critical=True
    )
    claim_award_allowed = (
        f"The provided award name '{award}' is one of the allowed major awards: "
        "National Book Award for Fiction; Booker Prize; Pulitzer Prize for Fiction; PEN/Faulkner Award for Fiction."
    )
    await evaluator.verify(
        claim=claim_award_allowed,
        node=award_name_leaf,
        additional_instruction="This is a pure string/category check; allow minor naming variants (case, punctuation, presence/absence of 'The'), but it must clearly correspond to one of the four allowed awards."
    )

    # 2) Award URLs provided (critical prerequisite within award section)
    award_urls_provided = evaluator.add_custom_node(
        result=bool(book.award_urls),
        id=f"book_{idx}_award_urls_provided",
        desc="Award verification URL(s) are provided",
        parent=award_section,
        critical=True
    )

    # 3) Award verification via URLs (critical per rubric: award_verification_url)
    award_verify_leaf = evaluator.add_leaf(
        id=f"book_{idx}_award_verification_url",
        desc="The provided award URL(s) confirm the 2024 award win for this book (fiction)",
        parent=award_section,
        critical=True
    )
    claim_award_verified = (
        f"The provided source(s) confirm that the book '{title}' by {author} won the {award} in {TARGET_YEAR}. "
        "If applicable, the page should explicitly indicate the Fiction category (for National Book Award and PEN/Faulkner). "
        "For Booker Prize and Pulitzer Prize for Fiction, it must be the 2024 winner in fiction."
    )
    await evaluator.verify(
        claim=claim_award_verified,
        node=award_verify_leaf,
        sources=book.award_urls,
        additional_instruction=(
            "Verify that the page explicitly supports the 2024 win for the specified book (and author) for the stated award. "
            "Allow reasonable naming variants (e.g., 'The Booker Prize'). If the URL is irrelevant, inaccessible, "
            "or does not explicitly confirm the 2024 win for the correct book, return False."
        )
    )

    # -------------------- Publication Section --------------------
    pub_section = evaluator.add_parallel(
        id=f"book_{idx}_publication_section",
        desc=f"Book #{idx} publication details validation",
        parent=book_node,
        critical=False
    )

    # 4) Publication URLs provided (critical prerequisite within publication section)
    pub_urls_provided = evaluator.add_custom_node(
        result=bool(book.publication_urls),
        id=f"book_{idx}_publication_urls_provided",
        desc="Publication verification URL(s) are provided",
        parent=pub_section,
        critical=True
    )

    # 5) Publisher name verification (critical)
    publisher_leaf = evaluator.add_leaf(
        id=f"book_{idx}_publisher_name",
        desc="The publisher's name is accurate per the provided publication URL(s)",
        parent=pub_section,
        critical=True
    )
    claim_publisher = f"The publisher of the book '{title}' is '{publisher}'."
    await evaluator.verify(
        claim=claim_publisher,
        node=publisher_leaf,
        sources=book.publication_urls,
        additional_instruction=(
            "Check the provided page(s) for the publisher/imprint. Treat recognized imprints as valid even if the parent "
            "company is different (e.g., 'Riverhead Books' under Penguin Random House). Minor naming variants are acceptable "
            "if clearly the same imprint/company."
        )
    )

    # 6) Publication date verification (critical)
    pub_date_leaf = evaluator.add_leaf(
        id=f"book_{idx}_publication_date",
        desc="The publication date (month day, year) is accurate per the provided publication URL(s)",
        parent=pub_section,
        critical=True
    )
    claim_pub_date = f"The publication date of the book '{title}' is '{pub_date}'."
    await evaluator.verify(
        claim=claim_pub_date,
        node=pub_date_leaf,
        sources=book.publication_urls,
        additional_instruction=(
            "Match the exact date for the edition implied by the answer. Allow reasonable date format variations "
            "(e.g., '1 August 2024' vs 'August 1, 2024'). If multiple editions/dates are shown, prefer the primary "
            "or first publication date that aligns with the answer. If the page only shows month/year but the answer has a full date, "
            "treat as unsupported unless the day can be inferred explicitly."
        )
    )

    # 7) Publication verification URL(s) confirm details (critical per rubric)
    pub_verify_leaf = evaluator.add_leaf(
        id=f"book_{idx}_publication_verification_url",
        desc="The provided publication URL(s) confirm both publisher and publication date for this book",
        parent=pub_section,
        critical=True
    )
    claim_pub_verify = (
        f"The provided source(s) confirm the publisher ('{publisher}') and publication date ('{pub_date}') for the book '{title}'."
    )
    await evaluator.verify(
        claim=claim_pub_verify,
        node=pub_verify_leaf,
        sources=book.publication_urls,
        additional_instruction=(
            "At least one provided URL should explicitly list both the publisher/imprint and the publication date matching the answer "
            "for the identified book. If none of the sources confirm both pieces of information, return False."
        )
    )

    # -------------------- Author Section --------------------
    # 8) Author name verification (critical)
    author_leaf = evaluator.add_leaf(
        id=f"book_{idx}_author_name",
        desc="The author's full name is accurate per provided sources",
        parent=book_node,
        critical=True
    )
    author_sources = _merge_urls(book.publication_urls, book.award_urls)
    claim_author = f"The author of the book '{title}' is '{author}'."
    await evaluator.verify(
        claim=claim_author,
        node=author_leaf,
        sources=author_sources if author_sources else None,
        additional_instruction=(
            "Verify the author's name on the provided page(s). Allow minor variants (middle initials, diacritics, case); "
            "the identity should be the same person."
        )
    )

    # -------------------- Book Club Selection (Non-Critical) --------------------
    club_leaf = evaluator.add_leaf(
        id=f"book_{idx}_book_club_selection",
        desc="Book club selection status is accurate (Reese's or Oprah's in 2024/2025, or explicitly not selected)",
        parent=book_node,
        critical=False
    )
    # Prefer dedicated club URLs if provided, else use any provided sources
    club_sources = _merge_urls(book.club_urls, book.award_urls, book.publication_urls)
    claim_club = (
        f"The book '{title}' has the following book club selection status as claimed: {club_sel}. "
        "If a selection is claimed (Reese's or Oprah's), at least one provided source should explicitly show that selection and the month/year. "
        "If 'Not selected by Reese's Book Club or Oprah's Book Club' is claimed, the sources should not show an official selection."
    )
    await evaluator.verify(
        claim=claim_club,
        node=club_leaf,
        sources=club_sources if club_sources else None,
        additional_instruction=(
            "Prefer official Reese's Book Club or Oprah's Book Club pages, or reputable announcements. "
            "For negative claims (not selected), if sources do not explicitly confirm non-selection, return False."
        )
    )

    # -------------------- Adaptation Status (Non-Critical) --------------------
    adapt_leaf = evaluator.add_leaf(
        id=f"book_{idx}_adaptation_status",
        desc="Screen adaptation status is accurate (platform/status if applicable)",
        parent=book_node,
        critical=False
    )
    adapt_sources = _merge_urls(book.adaptation_urls, book.award_urls, book.publication_urls)
    claim_adapt = (
        f"The screen adaptation status for '{title}' is as claimed: {adapt}. "
        "If an adaptation is claimed, at least one provided source should explicitly confirm it with platform or production details. "
        "If 'No adaptation information found' is claimed, the provided sources should not show credible adaptation information."
    )
    await evaluator.verify(
        claim=claim_adapt,
        node=adapt_leaf,
        sources=adapt_sources if adapt_sources else None,
        additional_instruction=(
            "Prefer official streamer/studio press pages or reputable trades (Variety, The Hollywood Reporter, Deadline). "
            "For negative claims, if sources do not explicitly confirm absence, return False."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    # Initialize evaluator with parallel root (per rubric)
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

    # Record ground-truth constraints (not for correctness of items, but for context)
    evaluator.add_ground_truth(
        {
            "allowed_awards": ALLOWED_AWARDS,
            "target_year": TARGET_YEAR,
            "requirements": {
                "fields": [
                    "award", "author", "publisher", "publication_date",
                    "book_club_selection", "adaptation_status",
                    "award_urls", "publication_urls"
                ],
                "books_required": 4
            }
        },
        gt_type="constraints"
    )

    # Extract structured books from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction",
    )

    # Normalize to exactly 4 items: take first 4, pad with empty if fewer
    books: List[BookEntry] = list(extracted.books[:4])
    while len(books) < 4:
        books.append(BookEntry())

    # Build book-level verification nodes
    for i in range(4):
        await verify_book(evaluator, root, books[i], i)

    # Return structured evaluation summary
    return evaluator.get_summary()