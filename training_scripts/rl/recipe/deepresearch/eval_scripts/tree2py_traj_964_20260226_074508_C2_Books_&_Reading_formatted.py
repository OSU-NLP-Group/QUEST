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
TASK_ID = "us_multi_award_books_2025"
TASK_DESCRIPTION = """
Identify book(s) published in the United States that won multiple major literary awards in 2025. The major literary awards to consider are: the National Book Award, the Pulitzer Prize, the Booker Prize, the Carnegie Medal for Excellence, and the National Book Critics Circle Award. For each book you identify, provide: (1) the book title and author name, (2) each award won (minimum of two awards from the specified list), (3) the specific category for each award (e.g., Fiction, Nonfiction, Poetry), (4) the publisher, and (5) a reference URL verifying each award win.
"""

ALLOWED_AWARD_PATTERNS = [
    "national book award",
    "pulitzer prize",
    "booker prize",
    "carnegie medal for excellence",
    "andrew carnegie medals for excellence",
    "carnegie medal",
    "national book critics circle award",
    "national book critics circle",
    "nbcc award"
]

MAX_BOOKS_TO_CONSIDER = 1  # per rubric: "at least one"; we evaluate only the first one provided


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AwardEntry(BaseModel):
    award_name: Optional[str] = None
    category: Optional[str] = None
    year: Optional[str] = None
    verification_urls: List[str] = Field(default_factory=list)


class BookItem(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    publication_country: Optional[str] = None
    awards: List[AwardEntry] = Field(default_factory=list)


class BooksExtraction(BaseModel):
    books: List[BookItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return """
    From the answer, extract up to 3 book entries. Each extracted book must be a book published in the United States and described as winning multiple (>= 2) major literary awards in the 2025 award cycle. Only consider awards from the following allowed list:
    - National Book Award
    - Pulitzer Prize
    - Booker Prize
    - Carnegie Medal for Excellence (also known as Andrew Carnegie Medals for Excellence in Fiction and Nonfiction)
    - National Book Critics Circle Award (NBCC)

    For each book, extract:
    - title: the book title
    - author: the author name
    - publisher: the publisher name (as stated)
    - publication_country: country of publication as stated (e.g., "United States", "U.S.", "USA")
    - awards: an array of award entries, where each entry contains:
        * award_name: the award name exactly as written in the answer
        * category: the specific category of the award (e.g., Fiction, Nonfiction, Poetry) if provided in the answer; else set to null
        * year: the award year or award cycle as stated (should be 2025); if not provided, set to null
        * verification_urls: a list of URLs explicitly provided in the answer that verify this award win (do not invent URLs; include only those present in the answer)

    Notes:
    - Extract only awards that are from the allowed list above.
    - Do not infer or add any information that is not explicitly present in the answer.
    - If any field is not provided for a book, set it to null (or empty list for URLs).
    - Ensure that verification_urls are actual URLs mentioned in the answer (plain links or markdown links).
    - If the answer lists more than 3 books, only extract the first 3 mentioned.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_text(s: Optional[str]) -> str:
    return (s or "").strip()


def is_us_publication(country: Optional[str]) -> bool:
    if not country:
        return False
    c = country.strip().lower()
    return any([
        "united states" in c,
        c in {"us", "u.s", "u.s.", "usa", "u.s.a"},
        "u.s." in c,
        "usa" in c,
    ])


def is_allowed_award_name(name: Optional[str]) -> bool:
    if not name:
        return False
    n = name.strip().lower()
    return any(pat in n for pat in ALLOWED_AWARD_PATTERNS)


def all_awards_from_allowed_list(awards: List[AwardEntry]) -> bool:
    return all(is_allowed_award_name(a.award_name) for a in awards if a and _safe_text(a.award_name))


def at_least_two_awards(awards: List[AwardEntry]) -> bool:
    return len([a for a in awards if a and _safe_text(a.award_name)]) >= 2


def categories_provided_for_all(awards: List[AwardEntry]) -> bool:
    if not awards:
        return False
    for a in awards:
        if not _safe_text(a.category):
            return False
    return True


def urls_present_for_each_award(awards: List[AwardEntry]) -> bool:
    if not awards:
        return False
    for a in awards:
        if not a.verification_urls or len(a.verification_urls) == 0:
            return False
    return True


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def verify_book_item(
    evaluator: Evaluator,
    parent_node,
    book: BookItem,
    book_index: int = 0,
) -> None:
    """
    Build the verification subtree for a single book item.
    All children under this node are critical, matching rubric semantics.
    """
    book_desc = f"Book #{book_index + 1} verification: meets all constraints"
    book_node = evaluator.add_parallel(
        id=f"book_{book_index}",
        desc=book_desc,
        parent=parent_node,
        critical=True
    )

    # book_title_and_author
    title_author_ok = bool(_safe_text(book.title)) and bool(_safe_text(book.author))
    evaluator.add_custom_node(
        result=title_author_ok,
        id=f"book_{book_index}_title_and_author",
        desc="Provides the book title and the author name.",
        parent=book_node,
        critical=True
    )

    # us_publication
    us_pub_ok = is_us_publication(book.publication_country)
    evaluator.add_custom_node(
        result=us_pub_ok,
        id=f"book_{book_index}_us_publication",
        desc="Book is published in the United States.",
        parent=book_node,
        critical=True
    )

    # publisher presence
    publisher_ok = bool(_safe_text(book.publisher))
    evaluator.add_custom_node(
        result=publisher_ok,
        id=f"book_{book_index}_publisher",
        desc="Provides the publisher name.",
        parent=book_node,
        critical=True
    )

    # Awards block
    awards_block = evaluator.add_parallel(
        id=f"book_{book_index}_awards_block",
        desc="Provides award-win information for the book, restricted to the specified award list, for the 2025 award cycle, including categories and verification URLs.",
        parent=book_node,
        critical=True
    )

    # award_minimum_count
    evaluator.add_custom_node(
        result=at_least_two_awards(book.awards),
        id=f"book_{book_index}_award_minimum_count",
        desc="Lists at least two award wins for the book from the specified award list.",
        parent=awards_block,
        critical=True
    )

    # awards_from_allowed_list
    evaluator.add_custom_node(
        result=all_awards_from_allowed_list(book.awards),
        id=f"book_{book_index}_awards_from_allowed_list",
        desc="All listed awards are from the allowed list: National Book Award, Pulitzer Prize, Booker Prize, Carnegie Medal for Excellence, National Book Critics Circle Award.",
        parent=awards_block,
        critical=True
    )

    # award_categories_provided
    evaluator.add_custom_node(
        result=categories_provided_for_all(book.awards),
        id=f"book_{book_index}_award_categories_provided",
        desc="For each listed award win, the specific award category is provided (e.g., Fiction, Nonfiction, Poetry).",
        parent=awards_block,
        critical=True
    )

    # verification_urls_per_award
    evaluator.add_custom_node(
        result=urls_present_for_each_award(book.awards),
        id=f"book_{book_index}_verification_urls_per_award",
        desc="For each listed award win, provides a reliable reference URL that verifies the win.",
        parent=awards_block,
        critical=True
    )

    # For each award, verify "win" and "2025 cycle" via the provided URLs.
    # These are critical checks under the awards block (as per rubric).
    # We parallelize URL verifications for efficiency.
    batch: List[tuple[str, List[str], Any, Optional[str]]] = []

    for aidx, award in enumerate(book.awards):
        # Skip empty award entries defensively
        if not _safe_text(award.award_name):
            continue

        # Create a small sequential node per award to host its checks
        per_award_node = evaluator.add_sequential(
            id=f"book_{book_index}_award_{aidx}",
            desc=f"Verification for award #{aidx + 1}: {_safe_text(award.award_name)}",
            parent=awards_block,
            critical=True
        )

        # Leaf: award is a win (not just nominated/shortlisted)
        win_leaf = evaluator.add_leaf(
            id=f"book_{book_index}_award_{aidx}_is_win",
            desc=f"'{_safe_text(book.title)}' by {_safe_text(book.author)} won the {_safe_text(award.award_name)} (not just nominated).",
            parent=per_award_node,
            critical=True
        )
        win_claim = (
            f"The provided source(s) explicitly confirm that the book '{_safe_text(book.title)}' "
            f"by {_safe_text(book.author)} won the {_safe_text(award.award_name)} (not merely nominated, finalist, or shortlisted)."
        )
        batch.append((
            win_claim,
            award.verification_urls,
            win_leaf,
            "Focus strictly on winner status. Do not accept nominee/shortlist/finalist/longlist. "
            "Allow minor variations in punctuation or casing for names and titles."
        ))

        # Leaf: award year/cycle is 2025
        yr_leaf = evaluator.add_leaf(
            id=f"book_{book_index}_award_{aidx}_year_2025",
            desc=f"The win for {_safe_text(award.award_name)} is in the 2025 award cycle.",
            parent=per_award_node,
            critical=True
        )
        yr_claim = (
            f"The provided source(s) indicate that the win of '{_safe_text(book.title)}' "
            f"by {_safe_text(book.author)} for {_safe_text(award.award_name)} is part of the 2025 award cycle (winners of 2025)."
        )
        batch.append((
            yr_claim,
            award.verification_urls,
            yr_leaf,
            "Accept mentions like '2025 winners', '2025 award', or 'award year 2025'. "
            "Do not accept 2024 or 2026."
        ))

    if batch:
        await evaluator.batch_verify(batch, majority_vote=True, num_trials=3)


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
    Evaluate an answer for the 'US books with multiple major awards in 2025' task.
    Only the first book provided in the answer is evaluated, aligning with the
    'at least one book' requirement.
    """
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

    # Extract structured data
    extracted = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction"
    )

    # Select up to the first book item for evaluation
    books = extracted.books[:MAX_BOOKS_TO_CONSIDER] if extracted and extracted.books else []

    if not books:
        # Create a failed critical node to reflect missing content according to rubric
        no_book_node = evaluator.add_custom_node(
            result=False,
            id="book_item_missing",
            desc="At least one identified book meets all stated constraints and required fields.",
            parent=root,
            critical=True
        )
        return evaluator.get_summary()

    # Build verification for the first book
    await verify_book_item(evaluator, root, books[0], 0)

    return evaluator.get_summary()