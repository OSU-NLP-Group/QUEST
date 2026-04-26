import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple, Set

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "books_2025_acclaimed_multi_lists"
TASK_DESCRIPTION = """
I'm compiling a reading list of the most acclaimed literary works from 2025 and want to focus on books that received multiple prestigious recognitions. Please identify four books published in 2025 that each appeared on at least two of the following major literary lists: the 2025 Booker Prize shortlist, the 2025 National Book Awards finalists (any category), The New York Times' 10 Best Books of 2025, or the 2025 Pulitzer Prize finalists. For each of the four books, provide the following information: (1) The complete book title, (2) The author's full name, (3) The publisher, and (4) All the major 2025 literary lists on which the book appeared (from the lists mentioned above). Please ensure that all information is accurate and verifiable through official sources.
"""

# Canonical list definitions and slugs
ALLOWED_LISTS: Dict[str, Dict[str, str]] = {
    "booker_shortlist": {
        "display": "Booker Prize shortlist",
        "year_display": "2025 Booker Prize shortlist",
    },
    "nba_finalists": {
        "display": "National Book Awards finalists",
        "year_display": "2025 National Book Awards finalists",
    },
    "nyt_10_best": {
        "display": "NYT 10 Best Books",
        "year_display": "The New York Times' 10 Best Books of 2025",
    },
    "pulitzer_finalists": {
        "display": "Pulitzer Prize finalists",
        "year_display": "2025 Pulitzer Prize finalists",
    },
}

ALLOWED_CANONICAL_NAMES: Set[str] = {v["display"] for v in ALLOWED_LISTS.values()}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BookListEvidence(BaseModel):
    list_name: Optional[str] = None  # One of the allowed list names if present
    urls: List[str] = Field(default_factory=list)  # URLs that support this membership


class BookExtract(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None  # If multiple authors were given, they may be combined as a single string
    publisher: Optional[str] = None
    list_memberships: List[BookListEvidence] = Field(default_factory=list)
    info_sources: List[str] = Field(default_factory=list)  # General/book/publisher official sources


class BooksExtraction(BaseModel):
    books: List[BookExtract] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers: normalization & claims                                             #
# --------------------------------------------------------------------------- #
def normalize_list_name(name: Optional[str]) -> Optional[str]:
    """
    Normalize varied list names into canonical display names used in ALLOWED_LISTS.
    Returns the canonical display name if recognized; otherwise None.
    """
    if not name:
        return None
    s = name.strip().lower()

    # Booker Prize shortlist (2025)
    if "booker" in s and "short" in s:
        return ALLOWED_LISTS["booker_shortlist"]["display"]

    # National Book Awards finalists (2025)
    if ("national" in s and "book" in s and "award" in s) and ("final" in s):
        return ALLOWED_LISTS["nba_finalists"]["display"]

    # NYT 10 Best Books of 2025
    if (("new york times" in s) or ("nyt" in s)) and ("10" in s) and ("best" in s):
        return ALLOWED_LISTS["nyt_10_best"]["display"]

    # Pulitzer Prize finalists (2025)
    if "pulitzer" in s and "final" in s:
        return ALLOWED_LISTS["pulitzer_finalists"]["display"]

    return None


def slug_for_display(display_name: str) -> Optional[str]:
    for slug, meta in ALLOWED_LISTS.items():
        if meta["display"] == display_name:
            return slug
    return None


def claim_for_membership(slug: str, book_title: str, book_author: Optional[str]) -> str:
    """
    Compose a claim string for verifying that a book appears on a specific 2025 list.
    """
    if slug == "booker_shortlist":
        return f"The book '{book_title}' appears on the 2025 Booker Prize shortlist."
    if slug == "nba_finalists":
        # NBA finalists can be any category; allow that flexibility
        if book_author:
            return f"The book '{book_title}' by {book_author} is a finalist for the 2025 National Book Awards (in any category)."
        else:
            return f"The book '{book_title}' is a finalist for the 2025 National Book Awards (in any category)."
    if slug == "nyt_10_best":
        return f"The book '{book_title}' appears on The New York Times' 10 Best Books of 2025."
    if slug == "pulitzer_finalists":
        return f"The book '{book_title}' is listed among the 2025 Pulitzer Prize finalists (in a relevant category)."
    # fallback
    return f"The book '{book_title}' appears on the specified 2025 list."


def collect_all_sources(book: BookExtract) -> List[str]:
    """
    Aggregate all possible URLs for verifying book's bibliographic info (title/author/publisher).
    """
    urls: List[str] = []
    urls.extend(book.info_sources or [])
    for mem in book.list_memberships or []:
        urls.extend(mem.urls or [])
    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u and (u not in seen):
            deduped.append(u)
            seen.add(u)
    return deduped


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return """
Extract up to four (4) books from the answer that the user claims are from 2025 and that appeared on multiple major lists. For each book, extract the following fields:

1) title: The complete book title as written in the answer (include subtitle if present).
2) author: The author's full name exactly as provided in the answer. If multiple authors are listed, keep them as a single string (e.g., "Alice Smith and Bob Jones").
3) publisher: The publisher name as given in the answer (e.g., "Penguin Random House", "Farrar, Straus and Giroux").
4) list_memberships: An array capturing each major 2025 list membership specified for this book in the answer, where each element contains:
   - list_name: Use one of ONLY the following canonical names if applicable; otherwise leave null:
        - "Booker Prize shortlist"
        - "National Book Awards finalists"
        - "NYT 10 Best Books"
        - "Pulitzer Prize finalists"
     If the answer uses a variant (e.g., "2025 Booker shortlist", "New York Times 10 Best of 2025"), normalize it to the closest canonical name above.
   - urls: All URLs cited in the answer that directly support the claim for this specific list membership (e.g., the official list page, official press release, or authoritative publication page). If none are provided in the answer, return an empty array.
5) info_sources: All other URLs provided in the answer that can help verify bibliographic details (e.g., publisher’s official book page, author/publisher announcements, or other authoritative references). Include only URLs explicitly present in the answer.

IMPORTANT RULES:
- Do not invent any data. Extract only what is explicitly present in the answer.
- Normalize list names to the canonical set above when applicable.
- Always include URLs exactly as present in the answer text (respect SPECIAL RULES FOR URL EXTRACTION).
- Return at most four books (if more are present, keep the first four mentioned).
"""


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_book(
    evaluator: Evaluator,
    parent_node,
    book: BookExtract,
    book_index: int,
) -> None:
    """
    Build the verification subtree for a single book.
    """
    # Parent node for this book (non-critical to allow partial credit across books)
    book_node = evaluator.add_parallel(
        id=f"book_{book_index+1}",
        desc=f"{['First','Second','Third','Fourth'][book_index]} book that appeared on at least two major 2025 literary lists",
        parent=parent_node,
        critical=False
    )

    title_val = (book.title or "").strip()
    author_val = (book.author or "").strip()
    publisher_val = (book.publisher or "").strip()

    all_sources = collect_all_sources(book)

    # ---- Title verification (critical) ----
    title_node = evaluator.add_leaf(
        id=f"book_{book_index+1}_title",
        desc="Provide the complete and accurate book title",
        parent=book_node,
        critical=True
    )
    title_claim = f"The book's complete title is '{title_val}'."
    await evaluator.verify(
        claim=title_claim,
        node=title_node,
        sources=all_sources if all_sources else None,
        additional_instruction="Verify that the page shows the book's title matching the claimed text (allow minor punctuation or casing differences). Prefer official publisher or list pages."
    )

    # ---- Author verification (critical) ----
    author_node = evaluator.add_leaf(
        id=f"book_{book_index+1}_author",
        desc="Provide the author's full name",
        parent=book_node,
        critical=True
    )
    if title_val and author_val:
        author_claim = f"The author(s) of the book '{title_val}' is/are '{author_val}'."
    elif author_val:
        author_claim = f"The book's author(s) is/are '{author_val}'."
    else:
        author_claim = "The book has the specified author(s)."
    await evaluator.verify(
        claim=author_claim,
        node=author_node,
        sources=all_sources if all_sources else None,
        additional_instruction="Verify that the author name(s) on the page match the claimed author(s). Allow minor spelling variants and middle initials. If multiple authors are listed, ensure they match as a set."
    )

    # ---- Publisher verification (critical) ----
    publisher_node = evaluator.add_leaf(
        id=f"book_{book_index+1}_publisher",
        desc="Provide the publisher name",
        parent=book_node,
        critical=True
    )
    if title_val and publisher_val:
        publisher_claim = f"The publisher of the book '{title_val}' is '{publisher_val}'."
    elif publisher_val:
        publisher_claim = f"The book's publisher is '{publisher_val}'."
    else:
        publisher_claim = "The book has the specified publisher."
    await evaluator.verify(
        claim=publisher_claim,
        node=publisher_node,
        sources=all_sources if all_sources else None,
        additional_instruction="Verify the publisher name; prefer the publisher's official book page or an authoritative listing. Accept imprint/parent company variations when clearly equivalent."
    )

    # ---- Lists count check (critical) ----
    # Create the critical 'lists' node per rubric; treat it as a container with a critical custom count check
    lists_container = evaluator.add_parallel(
        id=f"book_{book_index+1}_lists",
        desc="Identify at least two major 2025 literary lists (from: Booker Prize shortlist, National Book Awards finalists, NYT 10 Best Books, Pulitzer Prize finalists) on which this book appeared",
        parent=book_node,
        critical=True
    )

    # Normalize memberships to canonical display names and compute the set for counting
    normalized_memberships: List[Tuple[str, List[str]]] = []
    for mem in book.list_memberships or []:
        canon = normalize_list_name(mem.list_name)
        if canon and canon in ALLOWED_CANONICAL_NAMES:
            normalized_memberships.append((canon, mem.urls or []))

    unique_canon_lists = {nm[0] for nm in normalized_memberships}
    count_at_least_two = len(unique_canon_lists) >= 2

    evaluator.add_custom_node(
        result=count_at_least_two,
        id=f"book_{book_index+1}_lists_count",
        desc=f"At least two recognized 2025 lists identified for book #{book_index+1}",
        parent=lists_container,
        critical=True
    )

    # ---- Evidence checks for each claimed list membership (non-critical, for grounding) ----
    # We place these directly under the book node (non-critical), to avoid making every single list leaf critical.
    evidence_node = evaluator.add_parallel(
        id=f"book_{book_index+1}_lists_evidence",
        desc=f"Evidence verification for each claimed 2025 list membership for book #{book_index+1}",
        parent=book_node,
        critical=False
    )

    # Collapse duplicates by canonical list; aggregate URLs
    canon_to_urls: Dict[str, List[str]] = {}
    for canon_name, urls in normalized_memberships:
        canon_to_urls.setdefault(canon_name, [])
        for u in urls:
            if u and (u not in canon_to_urls[canon_name]):
                canon_to_urls[canon_name].append(u)

    for canon_name, urls in canon_to_urls.items():
        slug = slug_for_display(canon_name)
        if not slug:
            continue
        support_leaf = evaluator.add_leaf(
            id=f"book_{book_index+1}_list_support_{slug}",
            desc=f"Verify membership on '{canon_name}' (2025) for this book",
            parent=evidence_node,
            critical=False
        )
        claim = claim_for_membership(slug, title_val or "the book", author_val or None)
        # Include both membership-specific URLs and general info sources to loosen constraints
        membership_sources = (urls or []) + (book.info_sources or [])
        await evaluator.verify(
            claim=claim,
            node=support_leaf,
            sources=membership_sources if membership_sources else None,
            additional_instruction="Confirm the page corresponds to the 2025 list and that the book is named on that page (or in the official press release). Allow small title formatting differences."
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
    Evaluate an answer for the 2025 multi-recognitions books task and return the structured summary.
    """
    # Initialize evaluator (root is non-critical by design; we keep parallel aggregation across books)
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

    # Record canonical list info in the summary for transparency
    evaluator.add_custom_info(
        info={
            "allowed_lists": [
                ALLOWED_LISTS["booker_shortlist"]["year_display"],
                ALLOWED_LISTS["nba_finalists"]["year_display"],
                ALLOWED_LISTS["nyt_10_best"]["year_display"],
                ALLOWED_LISTS["pulitzer_finalists"]["year_display"],
            ],
            "policy_note": "Each book should appear on at least two of the allowed 2025 lists."
        },
        info_type="guidance",
        info_name="canonical_lists_2025"
    )

    # Extract up to four books and their details
    extracted = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction"
    )

    # Take first four books, pad with empty if fewer
    books: List[BookExtract] = list(extracted.books[:4])
    while len(books) < 4:
        books.append(BookExtract())

    # Build verification subtrees for each of the four books
    for idx in range(4):
        await verify_book(evaluator, root, books[idx], idx)

    # Return the evaluation summary (includes tree and extraction results)
    return evaluator.get_summary()