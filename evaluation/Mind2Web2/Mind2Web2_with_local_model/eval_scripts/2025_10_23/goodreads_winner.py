import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "goodreads_winner"
TASK_DESCRIPTION = """
I wish to gain a broader cultural perspective through reading. Please find two Goodreads Choice Awards for Fiction winners from 2011 to the present that are written by authors born in the Global South. Please list the book titles, authors, their country of birth, the year of the awards, and the links on Goodreads.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BookInfo(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    country_of_birth: Optional[str] = None
    award_year: Optional[int] = None
    goodreads_link: Optional[str] = None

class ExtractedBooks(BaseModel):
    books: List[BookInfo] = Field(default_factory=list)

class BookURLs(BaseModel):
    title: Optional[str] = None
    urls: List[str] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return """
    Extract information about Goodreads Choice Awards winners from the answer. For each book mentioned, extract:
    1. The book title
    2. The author's name
    3. The author's country of birth
    4. The year the book won the Goodreads Choice Award
    5. The Goodreads link for the book (if provided)

    If any of these pieces of information is missing, set the corresponding field to null.
    Extract all books mentioned in the answer, even if there are more than two.
    """

def prompt_extract_urls_for_book(book_title: str, author: str) -> str:
    return f"""
    Extract all URLs from the answer that are associated with the book titled "{book_title}" by {author}.
    Only extract URLs that are explicitly mentioned in the answer.
    These could be Goodreads links or other relevant URLs for this book or its author.
    Extract any URL that might contain information about this book, the author, their country of birth, 
    or the Goodreads Choice Award.
    """

# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_book(
    evaluator: Evaluator,
    parent_node,
    book: BookInfo,
    book_index: int,
    urls: List[str]
) -> None:
    """
    Verify all aspects of a single book entry.
    """
    # Create a parallel parent node for this book
    book_node = evaluator.add_parallel(
        id=f"book_{book_index}",
        desc=f"Evaluation of book {book_index + 1}: '{book.title}' by {book.author}",
        parent=parent_node,
        critical=False  # Non-critical to allow partial credit for each book
    )
    
    # Check if Goodreads URLs exist
    goodreads_urls = [url for url in urls if "goodreads.com" in url.lower()]
    if book.goodreads_link and "goodreads.com" in book.goodreads_link.lower() and book.goodreads_link not in goodreads_urls:
        goodreads_urls.append(book.goodreads_link)
    
    # Single comprehensive existence check for all required data
    completeness_check = evaluator.add_custom_node(
        result=(
            bool(book.title) and 
            bool(book.author) and 
            bool(book.country_of_birth) and
            bool(book.award_year) and
            bool(urls) and
            bool(goodreads_urls)
        ),
        id=f"book_{book_index}_completeness_check",
        desc="Check if all required information (title, author, country, year, URLs, and Goodreads link) is provided",
        parent=book_node,
        critical=True
    )
    
    # Dedicated critical node for year range check
    start_year = 2011
    current_year = datetime.utcnow().year
    year_range_check = evaluator.add_custom_node(
        result=bool(book.award_year and start_year <= book.award_year <= current_year),
        id=f"book_{book_index}_year_in_range",
        desc=f"Check if award year {book.award_year} is within valid range ({start_year}-{current_year})",
        parent=book_node,
        critical=True
    )
    
    # 1. Verify that the book is a Goodreads Choice Award winner
    winner_verification_node = evaluator.add_leaf(
        id=f"book_{book_index}_is_goodreads_winner",
        desc=f"Verify that '{book.title}' by {book.author} won a Goodreads Choice Award for Fiction in {book.award_year}",
        parent=book_node,
        critical=True
    )
    
    claim = f"The book '{book.title}' by {book.author} won the Goodreads Choice Award for Fiction in {book.award_year}."
    await evaluator.verify(
        claim=claim,
        node=winner_verification_node,
        sources=urls,
        additional_instruction="Verify that this book specifically won the Goodreads Choice Award for Fiction (not another category) in the exact year specified. The information must be clearly stated on the webpage."
    )
    
    # 2. Verify author's birth country
    birth_country_node = evaluator.add_leaf(
        id=f"book_{book_index}_author_birth_country",
        desc=f"Verify that {book.author} was born in {book.country_of_birth}",
        parent=book_node,
        critical=True
    )
    
    claim = f"The author {book.author} was born in {book.country_of_birth}."
    await evaluator.verify(
        claim=claim,
        node=birth_country_node,
        sources=urls,
        additional_instruction="Verify that this author was specifically born in the country mentioned. Look for explicit information about the author's birthplace or country of birth."
    )
    
    # 3. Verify country is in Global South
    global_south_node = evaluator.add_leaf(
        id=f"book_{book_index}_country_is_global_south",
        desc=f"Verify that {book.country_of_birth} is part of the Global South",
        parent=book_node,
        critical=True
    )
    
    claim = f"{book.country_of_birth} is part of the Global South."
    await evaluator.verify(
        claim=claim,
        node=global_south_node,
        additional_instruction="Verify whether this country is considered part of the Global South. The Global South typically includes countries in Africa, Latin America, Asia (excluding Japan, South Korea, Singapore), and Oceania (excluding Australia and New Zealand)."
    )
    
    # 4. Verify year with URL
    year_verification_node = evaluator.add_leaf(
        id=f"book_{book_index}_year_verification",
        desc=f"Verify that the award year {book.award_year} is correct",
        parent=book_node,
        critical=True
    )
    
    claim = f"The book '{book.title}' by {book.author} won the Goodreads Choice Award for Fiction in {book.award_year}."
    await evaluator.verify(
        claim=claim,
        node=year_verification_node,
        sources=urls,
        additional_instruction="Verify that this book specifically won the Goodreads Choice Award in the exact year specified. The information must be clearly stated on the webpage."
    )
    
    # 5. Verify the Goodreads link corresponds to the correct book
    link_verification_node = evaluator.add_leaf(
        id=f"book_{book_index}_goodreads_link_correct",
        desc=f"Verify that the Goodreads link corresponds to '{book.title}' by {book.author}",
        parent=book_node,
        critical=True
    )
    
    claim = f"This Goodreads URL is for the book '{book.title}' by {book.author}."
    await evaluator.verify(
        claim=claim,
        node=link_verification_node,
        sources=goodreads_urls,
        additional_instruction="Verify that the Goodreads URL specifically links to the book mentioned in the claim. The book title and author should be clearly visible or mentioned on the linked page."
    )

# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: openai.AsyncAzureOpenAI,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()
    
    # Initialize evaluator with parallel strategy for root
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
    
    # -------- 2. Extract structured info from the answer ---------------- #
    # Extract all books mentioned in the answer
    extracted_books = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=ExtractedBooks,
        extraction_name="extracted_books"
    )
    
    # -------- 3. Build verification tree -------------------------------- #
    # Process exactly 2 books as required by the task
    required_books = 2
    
    # Pad the list if we have fewer than required books
    while len(extracted_books.books) < required_books:
        extracted_books.books.append(BookInfo())
    
    # Verify each of the 2 required books
    for i in range(required_books):
        book = extracted_books.books[i]
        
        # Extract URLs for this specific book
        book_urls_info = await evaluator.extract(
            prompt=prompt_extract_urls_for_book(book.title or "", book.author or ""),
            template_class=BookURLs,
            extraction_name=f"book_{i}_urls"
        )
        
        urls = book_urls_info.urls or []
        if book.goodreads_link and book.goodreads_link not in urls:
            urls.append(book.goodreads_link)
        
        # Verify this book
        await verify_book(evaluator, root, book, i, urls)
    
    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()
