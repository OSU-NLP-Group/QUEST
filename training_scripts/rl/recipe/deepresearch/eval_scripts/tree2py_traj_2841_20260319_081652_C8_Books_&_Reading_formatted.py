import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "litfic_4books_march2026"
TASK_DESCRIPTION = """
Identify 4 literary fiction books that meet the following criteria as of March 2026:

1. Book A: Won the 2025 PEN/Faulkner Award for Fiction. The author must hold a graduate degree from Harvard University. The publisher must be headquartered in New York City.

2. Book B: Won the 2025 National Book Award for Fiction. The author must have been born in Amman, Jordan. The publisher must be headquartered in New York City.

3. Book C: Won the 2025 Booker Prize. The author must have been born in Montreal in 1974, with a Canadian mother and Hungarian father.

4. Book D: Published on April 22, 2025 by Berkley Books. The author must be Emily Henry, a #1 New York Times bestselling author.

For each book, provide:
- The book title
- The author's full name
- The publisher name
- Reference URLs that support the award information, author biographical details, and publisher information
"""

CUTOFF_YEAR = 2026
CUTOFF_MONTH = 3  # March
CUTOFF_HUMAN = "March 2026"


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class BookEntry(BaseModel):
    label: Optional[str] = None  # Expect one of A, B, C, D
    title: Optional[str] = None
    author_full_name: Optional[str] = None
    publisher_name: Optional[str] = None

    # Optional: short note about genre from the answer (free text)
    genre_note: Optional[str] = None

    # Evidence URLs grouped by purpose
    award_urls: List[str] = Field(default_factory=list)
    author_bio_urls: List[str] = Field(default_factory=list)
    publisher_urls: List[str] = Field(default_factory=list)

    # D-specific (can be present for others as null/empty)
    publication_date: Optional[str] = None
    publication_date_urls: List[str] = Field(default_factory=list)


class BooksExtraction(BaseModel):
    books: List[BookEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return """
    Extract up to 4 book entries from the answer, labeled explicitly as Book A, Book B, Book C, and Book D.
    For each book entry, extract the following fields exactly as stated in the answer:

    - label: One of "A", "B", "C", or "D". Use a single uppercase letter without "Book ".
    - title: The book title (string). If missing, null.
    - author_full_name: The full name of the author (string). If missing, null.
    - publisher_name: The publisher name (string). If missing, null.
    - genre_note: Any explicit mention of "literary fiction" or genre/classification wording tied to this book. If absent, null.

    - award_urls: List of URLs that support the book's award claim (e.g., winner announcements or official award pages). If none provided, return [].
    - author_bio_urls: List of URLs that support the author-specific constraints (e.g., birthplace, education, parent heritage, NYT status). If none provided, return [].
    - publisher_urls: List of URLs that support the publisher-specific information (e.g., publisher identity or headquarters). If none provided, return [].

    - publication_date: Only if Book D; otherwise null. Extract the stated publication date string as given in the answer, if present for D. If absent, null.
    - publication_date_urls: List of URLs that support the publication date (especially for Book D). If none provided, return [].

    Return a JSON object with a single field:
    {
      "books": [ BookEntry, BookEntry, ... ]
    }

    IMPORTANT:
    - Do not invent URLs. Only include URLs explicitly present in the answer text.
    - If a field is missing for a book, set it to null (or [] for URL lists).
    - Ensure labels are correctly mapped to A, B, C, D, matching the answer's labeling.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    return label.strip().upper().replace("BOOK ", "")


def _clean_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls or []:
        if not u:
            continue
        s = u.strip()
        if not s:
            continue
        # crude basic screening
        if not (s.startswith("http://") or s.startswith("https://")):
            # keep as-is if extractor didn't prepend; but avoid obviously invalid
            # Let downstream normalizer handle scheme if missing
            if s.startswith("www."):
                s = "http://" + s
            else:
                # skip non-URLs
                continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _combined_urls(book: BookEntry, include_award=True, include_author=True, include_publisher=True, include_date=True) -> List[str]:
    combo: List[str] = []
    if include_award:
        combo.extend(book.award_urls or [])
    if include_author:
        combo.extend(book.author_bio_urls or [])
    if include_publisher:
        combo.extend(book.publisher_urls or [])
    if include_date:
        combo.extend(book.publication_date_urls or [])
    return _clean_urls(combo)


def _map_by_label(extraction: BooksExtraction) -> Dict[str, BookEntry]:
    mapping: Dict[str, BookEntry] = {}
    for b in extraction.books:
        lbl = _normalize_label(b.label)
        if lbl in ("A", "B", "C", "D"):
            # keep first occurrence per label
            if lbl not in mapping:
                mapping[lbl] = b
    return mapping


def _all_urls_from_all_books(books: List[BookEntry]) -> List[str]:
    urls: List[str] = []
    for b in books:
        urls.extend(b.award_urls or [])
        urls.extend(b.author_bio_urls or [])
        urls.extend(b.publisher_urls or [])
        urls.extend(b.publication_date_urls or [])
    return _clean_urls(urls)


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def add_cutoff_verifications(evaluator: Evaluator, parent_node, urls: List[str], id_prefix: str) -> None:
    """
    For each provided URL, add a leaf node verifying that the page is not dated later than March 2026
    when such dating/versioning is available. If a page provides no visible date/version, treat as compliant.
    This is a non-critical partial-credit collection.
    """
    # Create a parallel container for the URL date checks
    cutoff_parent = evaluator.add_parallel(
        id=id_prefix,
        desc=f"All cited sources are not later than {CUTOFF_HUMAN} when a visible date/version exists",
        parent=parent_node,
        critical=False
    )

    for idx, url in enumerate(urls):
        leaf = evaluator.add_leaf(
            id=f"{id_prefix}_url_{idx+1}",
            desc=f"Source date/version no later than {CUTOFF_HUMAN}",
            parent=cutoff_parent,
            critical=False
        )
        claim = (
            f"This webpage shows a publication or last-updated date (if any visible) that is on or before {CUTOFF_HUMAN}. "
            f"If the page provides no visible date or version, consider this condition satisfied. "
            f"If the page clearly shows a date after {CUTOFF_HUMAN}, consider this claim false."
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=url,
            additional_instruction=f"Carefully check the page for any visible byline date, publication date, last updated timestamp, or footer/version date. "
                                   f"'On or before' means 2026-03-31 or earlier. If the page shows April 2026 or later, the claim is false."
        )


# --------------------------------------------------------------------------- #
# Book-specific verification builders                                          #
# --------------------------------------------------------------------------- #
async def verify_book_a(evaluator: Evaluator, parent_node, book: BookEntry) -> None:
    node = evaluator.add_parallel(
        id="Book_A_PEN_Faulkner",
        desc="Book A: PEN/Faulkner 2025 winner; author has Harvard graduate degree; publisher HQ is NYC; includes required fields and supporting URLs.",
        parent=parent_node,
        critical=False
    )

    # Required info presence (critical)
    evaluator.add_custom_node(
        result=bool(book.title and book.title.strip()),
        id="A_Title_Provided",
        desc="Provides a non-empty book title for Book A.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(book.author_full_name and book.author_full_name.strip()),
        id="A_Author_Full_Name_Provided",
        desc="Provides the author's full name for Book A.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(book.publisher_name and book.publisher_name.strip()),
        id="A_Publisher_Name_Provided",
        desc="Provides the publisher name for Book A.",
        parent=node,
        critical=True
    )

    # URL presence checks (critical)
    evaluator.add_custom_node(
        result=len(_clean_urls(book.award_urls)) > 0,
        id="A_URL_Award_Support",
        desc="At least one reference URL supports Book A's award claim.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_clean_urls(book.author_bio_urls)) > 0,
        id="A_URL_AuthorBio_Support",
        desc="At least one reference URL supports Book A's author education constraint (Harvard graduate degree).",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_clean_urls(book.publisher_urls)) > 0,
        id="A_URL_Publisher_Support",
        desc="At least one reference URL supports Book A's publisher HQ-in-NYC constraint.",
        parent=node,
        critical=True
    )

    # Literary fiction claim (critical)
    lf_leaf = evaluator.add_leaf(
        id="A_Literary_Fiction",
        desc="Book A is a literary fiction book.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The book titled "{book.title or ""}" is a literary fiction book/novel.',
        node=lf_leaf,
        sources=_combined_urls(book),
        additional_instruction="Use the provided pages to infer genre classification. Accept clear indications such as 'literary fiction', "
                               "'literary novel', or evidence from reputable reviews/award framing suggesting literary fiction. "
                               "If genre is clearly non-literary (e.g., primarily romance genre marketing), return false."
    )

    # Award verification (critical)
    award_leaf = evaluator.add_leaf(
        id="A_Award_Constraint",
        desc="Book A won the 2025 PEN/Faulkner Award for Fiction.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The book "{book.title or ""}" won the 2025 PEN/Faulkner Award for Fiction.',
        node=award_leaf,
        sources=_clean_urls(book.award_urls),
        additional_instruction="Ensure it says 'winner' (not longlist/shortlist/finalist) and the year is 2025."
    )

    # Author education constraint (critical)
    edu_leaf = evaluator.add_leaf(
        id="A_Author_Education_Constraint",
        desc="Book A's author holds a graduate degree from Harvard University.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The author {book.author_full_name or ""} holds a graduate degree (e.g., MA, MFA, JD, MD, MPH, MBA, MTS, EdM, PhD, etc.) from Harvard University.',
        node=edu_leaf,
        sources=_clean_urls(book.author_bio_urls),
        additional_instruction="Look for explicit graduate credential(s) from Harvard (not just attendance or undergrad)."
    )

    # Publisher HQ in NYC (critical)
    hq_leaf = evaluator.add_leaf(
        id="A_Publisher_HQ_Constraint",
        desc="Book A's publisher is headquartered in New York City.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The publisher {book.publisher_name or ""} is headquartered in New York City (NYC).',
        node=hq_leaf,
        sources=_clean_urls(book.publisher_urls),
        additional_instruction="Treat 'New York, NY' as NYC. Be careful: some publishers have multiple offices; confirm the HQ is in NYC."
    )


async def verify_book_b(evaluator: Evaluator, parent_node, book: BookEntry) -> None:
    node = evaluator.add_parallel(
        id="Book_B_National_Book_Award",
        desc="Book B: National Book Award 2025 winner; author born in Amman, Jordan; publisher HQ is NYC; includes required fields and supporting URLs.",
        parent=parent_node,
        critical=False
    )

    # Required info presence (critical)
    evaluator.add_custom_node(
        result=bool(book.title and book.title.strip()),
        id="B_Title_Provided",
        desc="Provides a non-empty book title for Book B.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(book.author_full_name and book.author_full_name.strip()),
        id="B_Author_Full_Name_Provided",
        desc="Provides the author's full name for Book B.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(book.publisher_name and book.publisher_name.strip()),
        id="B_Publisher_Name_Provided",
        desc="Provides the publisher name for Book B.",
        parent=node,
        critical=True
    )

    # URL presence checks (critical)
    evaluator.add_custom_node(
        result=len(_clean_urls(book.award_urls)) > 0,
        id="B_URL_Award_Support",
        desc="At least one reference URL supports Book B's award claim.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_clean_urls(book.author_bio_urls)) > 0,
        id="B_URL_AuthorBio_Support",
        desc="At least one reference URL supports Book B's author birthplace constraint.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_clean_urls(book.publisher_urls)) > 0,
        id="B_URL_Publisher_Support",
        desc="At least one reference URL supports Book B's publisher HQ-in-NYC constraint.",
        parent=node,
        critical=True
    )

    # Literary fiction claim (critical)
    lf_leaf = evaluator.add_leaf(
        id="B_Literary_Fiction",
        desc="Book B is a literary fiction book.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The book titled "{book.title or ""}" is a literary fiction book/novel.',
        node=lf_leaf,
        sources=_combined_urls(book),
        additional_instruction="Use the provided pages to infer genre classification; accept 'literary fiction' or equivalent evidence from credible sources."
    )

    # Award (National Book Award for Fiction, 2025) (critical)
    award_leaf = evaluator.add_leaf(
        id="B_Award_Constraint",
        desc="Book B won the 2025 National Book Award for Fiction.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The book "{book.title or ""}" won the 2025 National Book Award for Fiction.',
        node=award_leaf,
        sources=_clean_urls(book.award_urls),
        additional_instruction="Ensure it says 'Winner' for Fiction in 2025 (not longlist/finalist). Prefer nationalbook.org pages."
    )

    # Author birthplace Amman, Jordan (critical)
    birthplace_leaf = evaluator.add_leaf(
        id="B_Author_Birthplace_Constraint",
        desc="Book B's author was born in Amman, Jordan.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The author {book.author_full_name or ""} was born in Amman, Jordan.',
        node=birthplace_leaf,
        sources=_clean_urls(book.author_bio_urls),
        additional_instruction="Allow minor spelling variants; look for reliable bio pages."
    )

    # Publisher HQ in NYC (critical)
    hq_leaf = evaluator.add_leaf(
        id="B_Publisher_HQ_Constraint",
        desc="Book B's publisher is headquartered in New York City.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The publisher {book.publisher_name or ""} is headquartered in New York City (NYC).',
        node=hq_leaf,
        sources=_clean_urls(book.publisher_urls),
        additional_instruction="Treat 'New York, NY' as NYC; confirm HQ, not just an office."
    )


async def verify_book_c(evaluator: Evaluator, parent_node, book: BookEntry) -> None:
    node = evaluator.add_parallel(
        id="Book_C_Booker_Prize",
        desc="Book C: 2025 Booker Prize winner; author born in Montreal in 1974 with Canadian mother and Hungarian father; includes required fields and supporting URLs.",
        parent=parent_node,
        critical=False
    )

    # Required info presence (critical)
    evaluator.add_custom_node(
        result=bool(book.title and book.title.strip()),
        id="C_Title_Provided",
        desc="Provides a non-empty book title for Book C.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(book.author_full_name and book.author_full_name.strip()),
        id="C_Author_Full_Name_Provided",
        desc="Provides the author's full name for Book C.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(book.publisher_name and book.publisher_name.strip()),
        id="C_Publisher_Name_Provided",
        desc="Provides the publisher name for Book C.",
        parent=node,
        critical=True
    )

    # URL presence checks (critical)
    evaluator.add_custom_node(
        result=len(_clean_urls(book.award_urls)) > 0,
        id="C_URL_Award_Support",
        desc="At least one reference URL supports Book C's award claim.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_clean_urls(book.author_bio_urls)) > 0,
        id="C_URL_AuthorBio_Support_Birth",
        desc="At least one reference URL supports Book C's author birth details constraint (Montreal, 1974).",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_clean_urls(book.author_bio_urls)) > 0,
        id="C_URL_AuthorBio_Support_Heritage",
        desc="At least one reference URL supports Book C's author parent-heritage constraint (Canadian mother, Hungarian father).",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_clean_urls(book.publisher_urls)) > 0,
        id="C_URL_Publisher_Support",
        desc="At least one reference URL supports Book C's publisher identity.",
        parent=node,
        critical=True
    )

    # Literary fiction (critical)
    lf_leaf = evaluator.add_leaf(
        id="C_Literary_Fiction",
        desc="Book C is a literary fiction book.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The book titled "{book.title or ""}" is a literary fiction book/novel.',
        node=lf_leaf,
        sources=_combined_urls(book),
        additional_instruction="Use provided pages; accept clear 'literary fiction' evidence or established literary award framing."
    )

    # Booker Prize 2025 winner (critical)
    award_leaf = evaluator.add_leaf(
        id="C_Award_Constraint",
        desc="Book C won the 2025 Booker Prize.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The book "{book.title or ""}" won the 2025 Booker Prize.',
        node=award_leaf,
        sources=_clean_urls(book.award_urls),
        additional_instruction="Prefer thebookerprizes.com; ensure it says 'winner' (not short/longlist)."
    )

    # Author birth details (Montreal, 1974) (critical)
    birth_leaf = evaluator.add_leaf(
        id="C_Author_Birth_Details_Constraint",
        desc="Book C's author was born in Montreal in 1974.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The author {book.author_full_name or ""} was born in Montreal in 1974.',
        node=birth_leaf,
        sources=_clean_urls(book.author_bio_urls),
        additional_instruction="Allow 'Montréal' spelling and DOB formats (e.g., YYYY-MM-DD implies 1974)."
    )

    # Parent heritage (Canadian mother, Hungarian father) (critical)
    heritage_leaf = evaluator.add_leaf(
        id="C_Author_Parent_Heritage_Constraint",
        desc="Book C's author has a Canadian mother and a Hungarian father.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The author {book.author_full_name or ""} has a Canadian mother and a Hungarian father.',
        node=heritage_leaf,
        sources=_clean_urls(book.author_bio_urls),
        additional_instruction="Check credible bios/interviews. Exact phrasing may vary; essence must match."
    )

    # Publisher identity claim (non-explicitly required as a separate claim in rubric text, but URLs must support publisher info)
    pub_leaf = evaluator.add_leaf(
        id="C_Publisher_Identity_Claim",
        desc="Book C publisher identity is correctly stated.",
        parent=node,
        critical=True  # Make it critical to ensure publisher info is supported
    )
    await evaluator.verify(
        claim=f'The publisher of "{book.title or ""}" is {book.publisher_name or ""}.',
        node=pub_leaf,
        sources=_clean_urls(book.publisher_urls),
        additional_instruction="Verify the publisher imprint/brand shown for this specific title."
    )


async def verify_book_d(evaluator: Evaluator, parent_node, book: BookEntry) -> None:
    node = evaluator.add_parallel(
        id="Book_D_Berkley_Emily_Henry",
        desc="Book D: published April 22, 2025 by Berkley Books; author is Emily Henry; author is a #1 NYT bestselling author; includes required fields and supporting URLs.",
        parent=parent_node,
        critical=False
    )

    # Required info presence (critical)
    evaluator.add_custom_node(
        result=bool(book.title and book.title.strip()),
        id="D_Title_Provided",
        desc="Provides a non-empty book title for Book D.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(book.author_full_name and book.author_full_name.strip()),
        id="D_Author_Full_Name_Provided",
        desc="Provides the author's full name for Book D.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(book.publisher_name and book.publisher_name.strip()),
        id="D_Publisher_Name_Provided",
        desc="Provides the publisher name for Book D.",
        parent=node,
        critical=True
    )

    # URL presence checks (critical)
    evaluator.add_custom_node(
        result=len(_clean_urls(book.publication_date_urls)) > 0,
        id="D_URL_PublicationDate_Support",
        desc="At least one reference URL supports Book D's publication date (April 22, 2025).",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(len(_clean_urls(book.author_bio_urls)) > 0 or len(_clean_urls(book.publisher_urls)) > 0 or len(_clean_urls(book.publication_date_urls)) > 0),
        id="D_URL_Author_Identity_Support",
        desc="At least one reference URL supports that the book's author is Emily Henry.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_clean_urls(book.author_bio_urls)) > 0,
        id="D_URL_Author_Status_Support",
        desc="At least one reference URL supports that Emily Henry is a #1 New York Times bestselling author.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_clean_urls(book.publisher_urls)) > 0,
        id="D_URL_Publisher_Support",
        desc="At least one reference URL supports Book D's publisher identity (Berkley Books).",
        parent=node,
        critical=True
    )

    # Literary fiction (critical)
    lf_leaf = evaluator.add_leaf(
        id="D_Literary_Fiction",
        desc="Book D is a literary fiction book.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The book titled "{book.title or ""}" is a literary fiction book/novel.',
        node=lf_leaf,
        sources=_combined_urls(book),
        additional_instruction="Use provided pages; if evidence clearly shows primary marketing/category is not literary fiction (e.g., contemporary romance), mark false."
    )

    # Author identity is Emily Henry (critical)
    author_leaf = evaluator.add_leaf(
        id="D_Author_Constraint",
        desc="Book D's author is Emily Henry.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The author of "{book.title or ""}" is Emily Henry.',
        node=author_leaf,
        sources=_combined_urls(book, include_award=False),  # award URLs not relevant here
        additional_instruction="Prefer official publisher listing or reputable retailer pages for the specific title."
    )

    # Author status: #1 NYT bestselling author (critical)
    status_leaf = evaluator.add_leaf(
        id="D_Author_Status_Constraint",
        desc="Emily Henry is a #1 New York Times bestselling author.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Emily Henry is a #1 New York Times bestselling author.",
        node=status_leaf,
        sources=_clean_urls(book.author_bio_urls),
        additional_instruction="Look for explicit '#1 New York Times bestselling author' or evidence of reaching #1 on the NYT list."
    )

    # Publication date April 22, 2025 (critical)
    pubdate_leaf = evaluator.add_leaf(
        id="D_Publication_Date_Constraint",
        desc="Book D was published on April 22, 2025.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The book "{book.title or ""}" was published on April 22, 2025.',
        node=pubdate_leaf,
        sources=_clean_urls(book.publication_date_urls),
        additional_instruction="Accept reasonable date formatting variants and timezone-neutral representations; the calendar date must be 2025-04-22."
    )

    # Publisher is Berkley Books (critical)
    publisher_leaf = evaluator.add_leaf(
        id="D_Publisher_Constraint",
        desc="Book D was published by Berkley Books.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The book "{book.title or ""}" was published by Berkley Books.',
        node=publisher_leaf,
        sources=_clean_urls(book.publisher_urls) or _clean_urls(book.publication_date_urls),
        additional_instruction="Verify imprint accurately; Berkley is an imprint of Penguin Random House, but the page for the title should show Berkley."
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
    """
    Evaluate an answer for the '4 literary fiction books with constraints (as of March 2026)' task.
    """
    # Initialize evaluator (Root set to PARALLEL; non-critical to allow partial scoring across sections)
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

    # 1) Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction",
    )

    # 2) Mapping and presence checks
    by_label = _map_by_label(extracted)
    # Ensure placeholders exist for each label to continue building the tree
    a = by_label.get("A", BookEntry(label="A"))
    b = by_label.get("B", BookEntry(label="B"))
    c = by_label.get("C", BookEntry(label="C"))
    d = by_label.get("D", BookEntry(label="D"))

    # All four books present (critical)
    def _has_all_four_distinct() -> bool:
        labels_ok = all(lbl in by_label for lbl in ("A", "B", "C", "D"))
        if not labels_ok:
            return False
        titles = []
        for bk in [a, b, c, d]:
            if not (bk.title and bk.title.strip()):
                return False
            titles.append((bk.title or "").strip().lower())
        return len(set(titles)) == len(titles)

    evaluator.add_custom_node(
        result=_has_all_four_distinct(),
        id="All_Four_Books_Present",
        desc="Response includes four distinct, clearly labeled entries for Book A, Book B, Book C, and Book D.",
        parent=root,
        critical=True,
    )

    # 3) As-of-March-2026 source-date checks (non-critical, partial credit)
    # Build a single container node and add one child per URL
    all_urls = _all_urls_from_all_books([a, b, c, d])
    if all_urls:
        await add_cutoff_verifications(evaluator, root, all_urls, id_prefix="As_Of_March_2026")
    else:
        # If no URLs at all, still add the parent node (with no children) to reflect in tree
        evaluator.add_parallel(
            id="As_Of_March_2026",
            desc=f"All cited sources are not later than {CUTOFF_HUMAN} when a visible date/version exists",
            parent=root,
            critical=False
        )

    # 4) Per-book verification groups
    await verify_book_a(evaluator, root, a)
    await verify_book_b(evaluator, root, b)
    await verify_book_c(evaluator, root, c)
    await verify_book_d(evaluator, root, d)

    # Optional: record ground-truth constraints for transparency
    evaluator.add_ground_truth({
        "constraints": {
            "A": {
                "award": "2025 PEN/Faulkner Award for Fiction",
                "author_education": "Harvard graduate degree",
                "publisher_hq": "NYC",
            },
            "B": {
                "award": "2025 National Book Award for Fiction",
                "author_birthplace": "Amman, Jordan",
                "publisher_hq": "NYC",
            },
            "C": {
                "award": "2025 Booker Prize",
                "author_birth_details": "Born in Montreal in 1974",
                "author_parent_heritage": "Canadian mother, Hungarian father",
            },
            "D": {
                "publication_date": "April 22, 2025",
                "publisher": "Berkley Books",
                "author": "Emily Henry",
                "author_status": "#1 New York Times bestselling author",
            }
        },
        "cutoff": CUTOFF_HUMAN
    })

    # 5) Return summary
    return evaluator.get_summary()