import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "award_2024_special_editions_ca"
TASK_DESCRIPTION = (
    "I'm a book collector in California looking to purchase special edition copies of award-winning fiction from 2024. "
    "I want to buy books that won major literary awards (Pulitzer Prize for Fiction, National Book Award for Fiction, or Booker Prize) "
    "and are available in special edition formats with distinctive features like sprayed edges, signed copies, exclusive covers, or collector's editions. "
    "Find at least 3 different fiction books that won one of these major literary awards in 2024 and are currently available in special edition format at Barnes & Noble or Books-A-Million. "
    "The bookstore chain must have at least one physical store location in California where I could potentially purchase these books. "
    "For each book, provide: the book title, author name, which major literary award it won in 2024, special edition feature(s) available, retailer name (Barnes & Noble or Books-A-Million), and reference URL showing the special edition availability."
)

ALLOWED_MAJOR_AWARDS = [
    "Pulitzer Prize for Fiction",
    "National Book Award for Fiction",
    "Booker Prize"
]

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class BookItem(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    award_name: Optional[str] = None
    award_year: Optional[str] = None  # Keep as string for flexibility
    special_features: List[str] = Field(default_factory=list)
    retailer: Optional[str] = None  # Barnes & Noble or Books-A-Million
    reference_url: Optional[str] = None  # Product page URL showing special edition availability
    # Optional supporting URLs if included in the answer:
    award_source_urls: List[str] = Field(default_factory=list)        # e.g., official award page or news
    retailer_store_urls: List[str] = Field(default_factory=list)      # e.g., store locator or CA store page(s)


class BooksExtraction(BaseModel):
    books: List[BookItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return """
    Extract up to 5 book entries from the answer that claim to be special edition copies of award-winning fiction from 2024 at major US bookstore chains.
    For each book, extract the following fields exactly as stated in the answer:
    - title: The book title.
    - author: The author name.
    - award_name: The major literary award the book won (e.g., "Pulitzer Prize for Fiction", "National Book Award for Fiction", or "Booker Prize").
    - award_year: The year associated with the award if mentioned (e.g., "2024"); if not mentioned, set to null.
    - special_features: A list of special edition feature(s) mentioned in the answer (e.g., "sprayed edges", "signed copy", "exclusive cover", "collector's edition").
    - retailer: The retailer name, must be either Barnes & Noble or Books-A-Million, as presented in the answer.
    - reference_url: The URL that shows the special edition availability (typically the product page on the retailer's website). If multiple URLs are present, choose the most directly relevant product page.
    - award_source_urls: Any additional URLs in the answer that support the award claim (e.g., an official prize page or a news article). If none are provided, return an empty list.
    - retailer_store_urls: Any URLs in the answer that support the retailer having physical store(s) in California (e.g., store locator or a California stores page). If none are provided, return an empty list.

    Return a JSON object in this schema:
    {
      "books": [
        {
          "title": ...,
          "author": ...,
          "award_name": ...,
          "award_year": ...,
          "special_features": [...],
          "retailer": ...,
          "reference_url": ...,
          "award_source_urls": [...],
          "retailer_store_urls": [...]
        },
        ...
      ]
    }

    Rules:
    - Only extract what is explicitly present in the answer. Do not infer or invent missing information.
    - Ensure URLs are fully qualified. If a URL is missing the protocol, prepend http://
    - If any field is not provided in the answer, set it to null (or [] for list fields).
    - Preserve the original casing of text fields.
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def normalize_retailer(retailer: Optional[str]) -> str:
    """Normalize retailer name for downstream logic."""
    if not retailer:
        return "Unknown"
    r = retailer.strip().lower()
    if "barnes" in r or "b&n" in r or "barnes & noble" in r:
        return "Barnes & Noble"
    if "books-a-million" in r or "booksamillion" in r or "bam" in r:
        return "Books-A-Million"
    return retailer.strip()


def retailer_ca_source_urls(retailer: str, provided_urls: Optional[List[str]] = None) -> List[str]:
    """
    Provide URLs to verify whether the retailer has at least one physical store location in California.
    Prefer URLs provided in the answer. Otherwise, fall back to well-known locator pages.
    """
    provided_urls = provided_urls or []
    if provided_urls:
        return provided_urls

    norm = normalize_retailer(retailer)
    if norm == "Barnes & Noble":
        # Provide multiple URLs to increase robustness
        return [
            "https://stores.barnesandnoble.com/stores?state=CA",
            "https://stores.barnesandnoble.com/store-locator?state=CA",
            "https://stores.barnesandnoble.com/"
        ]
    if norm == "Books-A-Million":
        return [
            "https://www.booksamillion.com/storefinder"
        ]
    return []


def first_k_books(books: List[BookItem], k: int = 3) -> List[BookItem]:
    trimmed = books[:k]
    while len(trimmed) < k:
        trimmed.append(BookItem())
    return trimmed


# --------------------------------------------------------------------------- #
# Verification for each book                                                  #
# --------------------------------------------------------------------------- #
async def verify_book(
    evaluator: Evaluator,
    parent_node,
    book: BookItem,
    idx: int
) -> None:
    book_idx = idx + 1
    book_node = evaluator.add_parallel(
        id=f"Book_{book_idx}",
        desc=f"{['First','Second','Third','Fourth','Fifth'][idx] if idx < 5 else f'Book #{book_idx}'} award-winning fiction book with special edition availability",
        parent=parent_node,
        critical=False
    )

    # Output completeness (critical) – gate other checks
    has_title = bool(book.title and book.title.strip())
    has_author = bool(book.author and book.author.strip())
    has_award = bool(book.award_name and book.award_name.strip())
    has_features = bool(book.special_features and any(f.strip() for f in book.special_features))
    has_retailer = bool(book.retailer and book.retailer.strip())
    has_ref_url = bool(book.reference_url and book.reference_url.strip())
    evaluator.add_custom_node(
        result=has_title and has_author and has_award and has_features and has_retailer and has_ref_url,
        id=f"Book_{book_idx}_Output_Information",
        desc="All required information is provided: book title, author name, award won, special edition feature(s), retailer name, and reference URL showing availability",
        parent=book_node,
        critical=True
    )

    # Fiction genre (critical) – verify via product page
    fiction_leaf = evaluator.add_leaf(
        id=f"Book_{book_idx}_Fiction_Genre",
        desc="The book is in the fiction category",
        parent=book_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The product page indicates that the book '{book.title or ''}' is a work of fiction (e.g., a novel) rather than nonfiction.",
        node=fiction_leaf,
        sources=book.reference_url if book.reference_url else None,
        additional_instruction="Look for category tags, breadcrumbs, or descriptors like 'Fiction', 'Literary Fiction', 'General Fiction', 'Novels', or similar on the page. "
                               "Allow reasonable synonyms and consider 'novel' as fiction."
    )

    # Award verification (critical) – verify winner status and year from sources
    award_leaf = evaluator.add_leaf(
        id=f"Book_{book_idx}_Award",
        desc="The book won a major 2024 literary award in fiction (Pulitzer Prize for Fiction, National Book Award for Fiction, or Booker Prize)",
        parent=book_node,
        critical=True
    )
    award_sources: List[str] = []
    if book.reference_url:
        award_sources.append(book.reference_url)
    if book.award_source_urls:
        award_sources.extend(book.award_source_urls)
    # Build a conservative claim focused on winner + year
    award_year_text = book.award_year.strip() if (book.award_year and book.award_year.strip()) else "2024"
    await evaluator.verify(
        claim=f"The book '{book.title or ''}' won the {award_year_text} {book.award_name or ''}. "
              f"It is a winner (not merely shortlisted, longlisted, or a finalist).",
        node=award_leaf,
        sources=award_sources if award_sources else None,
        additional_instruction="Verify that the page(s) explicitly state 'Winner' or otherwise clearly indicate the book won the specified 2024 award. "
                               "Do not accept 'shortlisted', 'longlisted', or 'finalist' as 'won'. "
                               "Allow reasonable variations of award naming (e.g., 'The Booker Prize' vs 'Booker Prize')."
    )

    # Special edition verification (critical) – verify distinctive features on product page
    special_leaf = evaluator.add_leaf(
        id=f"Book_{book_idx}_Special_Edition",
        desc="The book is available in a special edition format with at least one distinguishing feature (sprayed edges, signed copy, exclusive cover, or collector's edition)",
        parent=book_node,
        critical=True
    )
    features_text = ", ".join(book.special_features) if book.special_features else ""
    await evaluator.verify(
        claim=f"The product page offers a special edition of '{book.title or ''}' with at least one distinguishing feature such as: {features_text}.",
        node=special_leaf,
        sources=book.reference_url if book.reference_url else None,
        additional_instruction="Look for labels or descriptors like 'Exclusive Edition', 'Barnes & Noble Exclusive', 'Books-A-Million Exclusive', 'Signed Book', 'Sprayed Edges', 'Collector's Edition', "
                               "'Exclusive Cover', or synonyms. The page should clearly present at least one such special feature."
    )

    # Retailer verification (critical) – ensure page belongs to B&N or BAM
    retailer_leaf = evaluator.add_leaf(
        id=f"Book_{book_idx}_Retailer",
        desc="The special edition is available at Barnes & Noble or Books-A-Million",
        parent=book_node,
        critical=True
    )
    retailer_norm = normalize_retailer(book.retailer)
    await evaluator.verify(
        claim=f"The product page belongs to the retailer '{retailer_norm}', and the special edition is available for purchase through this retailer.",
        node=retailer_leaf,
        sources=book.reference_url if book.reference_url else None,
        additional_instruction="Verify that the domain and branding correspond to the stated retailer. "
                               "Accept 'barnesandnoble.com' for Barnes & Noble, and 'booksamillion.com' for Books-A-Million."
    )

    # California physical store presence (critical) – verify via store locator or equivalent evidence
    ca_store_leaf = evaluator.add_leaf(
        id=f"Book_{book_idx}_California_Store",
        desc="The retailer has at least one physical store location in California",
        parent=book_node,
        critical=True
    )
    ca_sources = retailer_ca_source_urls(retailer_norm, book.retailer_store_urls)
    await evaluator.verify(
        claim=f"{retailer_norm} has at least one physical store location in California.",
        node=ca_store_leaf,
        sources=ca_sources if ca_sources else None,
        additional_instruction="Check the store locator or locations page(s) to confirm that at least one store is in California. "
                               "Evidence may include a state list containing 'California', a filter set to CA, or specific store addresses in California."
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
    # Initialize evaluator with a parallel root (non-critical to allow partial credit across books)
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

    # Add helpful context info
    evaluator.add_custom_info(
        info={"allowed_major_awards": ALLOWED_MAJOR_AWARDS, "required_retailers": ["Barnes & Noble", "Books-A-Million"], "min_books": 3},
        info_type="task_policy",
        info_name="evaluation_requirements"
    )

    # Extract structured book data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction"
    )

    # Use first 3 books (pad with placeholders if fewer)
    books = first_k_books(extracted.books, 3)

    # Build verification subtree for each of the 3 requested books
    main_node = evaluator.add_parallel(
        id="Award_Winning_Fiction_Books_Special_Editions",
        desc="Find at least 3 fiction books that won major literary awards in 2024 and are available in special edition formats at major US bookstore chains with physical locations in California",
        parent=root,
        critical=False  # Keep parent non-critical to avoid critical-children constraint and allow partial credit
    )

    # Verify each book node
    for i, bk in enumerate(books):
        await verify_book(evaluator, main_node, bk, i)

    # Return final evaluation summary
    return evaluator.get_summary()