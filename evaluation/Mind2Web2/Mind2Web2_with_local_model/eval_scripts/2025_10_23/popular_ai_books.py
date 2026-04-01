import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.eval_toolkit import create_evaluator, Extractor, Verifier
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.evaluator import Evaluator

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "popular_ai_books"
TASK_DESCRIPTION = """
I am interested in reading popular science books about Artificial Intelligence (AI) or closely related topics, written (or co-written) by recipients of the prestigious Turing Award. Please help me identify five suitable books that are specifically intended for general readers, rather than textbooks aimed at students or academic reference manuals. For each selected book, please provide the title, the name(s) of the author(s), the year in which the author(s) received the Turing Award, and a link to its Goodreads page.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BookTitles(BaseModel):
    """Model for extracting just the book titles."""
    titles: List[str] = Field(default_factory=list)

class TuringUrls(BaseModel):
    turing_award_urls : List[str] = Field(default_factory=list)

class TuringAuthors(BaseModel):
    """Model for extracting URLs related to Turing Award recipients."""
    author_name: Optional[str]
    award_year: Optional[str]
    
class BookRecommendation(BaseModel):
    """Model for a single book recommendation."""
    title: Optional[str]
    authors: Optional[List[str]] = Field(default_factory=list)
    turing_authors: Optional[List[TuringAuthors]] = Field(default_factory=list)
    goodreads_url: Optional[str]


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_book_titles() -> str:
    return """
    Extract just the titles of the books recommended in the answer.
    Return the book titles exactly as they appear in the answer.
    """

def prompt_extract_book_details(book_title: str) -> str:
    return f"""
    Extract the following information about the book titled "{book_title}":
    
    1. The full title of the book (should match "{book_title}")
    2. The name(s) of the author(s)
    3. Which author(s) recieved turing award, and the year they received it in
    4. The Goodreads URL for the book, if provided
    
    Only extract information related to this specific book. If any information is missing, set the corresponding field to null.
    """

def prompt_extract_turing_award_urls(turing_winner: str, turing_year) -> str:
    return f"""
    Extract the all URLs that support that {turing_winner} won the Turing Award in {turing_year}, or similar phrasing. Only mention those that mention it explicitly.
    """

# --------------------------------------------------------------------------- #
# Verification functions for individual components                            #
# --------------------------------------------------------------------------- #
async def verify_turing_author(
        evaluator: Evaluator,
        parent_node,
        turing_author: TuringAuthors,
        book_authors: List[str],
        turing_urls: List[str],
        book_index: int,
        author_index: int,
) -> None:
    """
    Verify a single Turing author - both that they won the award and that they're an author of the book.
    """
    # Container for this Turing author verification
    turing_author_node = evaluator.add_parallel(
        id=f"book_{book_index}_turing_author_{author_index}",
        desc=f"Turing author #{author_index+1}: {turing_author.author_name or 'Unknown'}",
        parent=parent_node,
        critical=True
    )
    
    # Check existence of required data
    data_exists_check = evaluator.add_custom_node(
        result=bool(turing_author.author_name and turing_author.award_year and turing_urls),
        id=f"book_{book_index}_turing_author_{author_index}_data_exists",
        desc=f"Turing author data and sources exist",
        parent=turing_author_node,
        critical=True
    )
    
    # Verify this person won the Turing Award
    turing_award_verify = evaluator.add_leaf(
        id=f"book_{book_index}_turing_author_{author_index}_award_verify",
        desc=f"{turing_author.author_name} won Turing Award in {turing_author.award_year}",
        parent=turing_author_node,
        critical=True
    )
    
    award_claim = f"{turing_author.author_name} received the Turing Award in {turing_author.award_year}."
    
    await evaluator.verify(
        claim=award_claim,
        node=turing_award_verify,
        sources=turing_urls,
        additional_instruction="Verify whether that the source supports the recipient of the turing award won the prize in specified year. The Turing Award is the highest distinction in computer science, often referred to as the 'Nobel Prize of Computing'."
    )
    
    # Verify this Turing winner is actually an author of the book
    is_author_verify = evaluator.add_leaf(
        id=f"book_{book_index}_turing_author_{author_index}_is_author",
        desc=f"{turing_author.author_name} is an author of this book",
        parent=turing_author_node,
        critical=True
    )
    
    authors_str = ", ".join(book_authors) if book_authors else "No authors listed"
    authorship_claim = f"{turing_author.author_name} is one of the authors of the book. The book's authors are: {authors_str}"
    
    await evaluator.verify(
        claim=authorship_claim,
        node=is_author_verify,
        sources=None,  # This is a simple verification without URL
        additional_instruction="Verify if the Turing Award winner's name appears in the list of book authors. Allow for reasonable variations in name format (e.g., with/without middle names, initials)."
    )

async def verify_single_book(
        evaluator: Evaluator,
        parent_node,
        book: BookRecommendation,
        book_index: int,
) -> None:
    """
    Perform complete verification for a single book recommendation.
    """
    book_node = evaluator.add_parallel(
        id=f"book_{book_index}",
        desc=f"Book recommendation #{book_index+1}: {book.title if book.title else 'Unknown'}",
        parent=parent_node
    )
    
    # 1. Book completeness verification

    # Check existence of required fields
    completeness_check = evaluator.add_custom_node(
        result=bool(book.title and book.authors and book.turing_authors and book.goodreads_url and "goodreads.com" in (book.goodreads_url or "").lower()),
        id=f"book_{book_index}_completeness_check",
        desc=f"Book #{book_index+1} has title, author(s), Turing Award info, and Goodreads URL",
        parent=book_node,
        critical=True
    )
    
    # Verify the Goodreads URL corresponds to the book
    goodreads_verify = evaluator.add_leaf(
        id=f"book_{book_index}_goodreads_verify",
        desc=f"Goodreads URL corresponds to the book and authors",
        parent=book_node,
        critical=True
    )
    
    if book.title and book.authors:
        authors_str = ", ".join(book.authors)
        claim = f"The book {book.title} has exactly authors {authors_str}, and no one else. This is directly supported on the goodreads page given for {book.title}."
    else:
        claim = "Book information is missing"
    
    await evaluator.verify(
        claim=claim,
        node=goodreads_verify,
        sources=book.goodreads_url,
        additional_instruction="The page should contain the book title and author information that matches what was claimed, and it should be obvious the book is from goodreads."
    )
    
    # 2. Turing Award verification - now for ALL Turing authors
    if book.turing_authors:
        turing_container = evaluator.add_parallel(
            id=f"book_{book_index}_turing_authors",
            desc=f"All Turing Award authors verification",
            parent=book_node,
            critical=True  # Changed to critical
        )
        
        # Verify each Turing author
        for author_idx, turing_author in enumerate(book.turing_authors):
            # Extract URLs for this specific Turing author
            if turing_author.author_name and turing_author.award_year:
                turing_urls_result = await evaluator.extract(
                    prompt=prompt_extract_turing_award_urls(turing_author.author_name, turing_author.award_year),
                    template_class=TuringUrls,
                    extraction_name=f"turing_urls_book_{book_index}_author_{author_idx}"
                )
                turing_urls = turing_urls_result.turing_award_urls
            else:
                turing_urls = []
            
            await verify_turing_author(
                evaluator,
                turing_container,
                turing_author,
                book.authors,
                turing_urls,
                book_index,
                author_idx
            )
    else:
        # No Turing authors listed - create a failing node
        no_turing_node = evaluator.add_custom_node(
            result=False,
            id=f"book_{book_index}_no_turing_authors",
            desc="No Turing Award authors listed for this book",
            parent=book_node,
            critical=True
        )
    
    # 3. Book type verification (general audience) - directly under book node
    book_type_verify = evaluator.add_leaf(
        id=f"book_{book_index}_general_audience",
        desc=f"Book is for general readers, not a textbook",
        parent=book_node,
        critical=True  # Changed to critical
    )
    
    claim = (
        f"The book '{book.title}' is suitable for general readers. "
        "It is not a textbook designed explicitly for use by students in academic courses, "
        "nor is it an academic reference manual aimed primarily at researchers or professionals "
        "with specialized technical knowledge."
    )
    
    await evaluator.verify(
        claim=claim,
        node=book_type_verify,
        sources=book.goodreads_url,
        additional_instruction="The description of the book may not explicitly state that it is intended for general readers. Therefore, you should infer this based on the content. If the book description lacks heavy technical language, complex terminology, or highly specialized details aimed at a niche audience, it is likely meant for a general audience. Books that are not dense with academic jargon, industry-specific terms, or complex theoretical concepts are typically suitable for readers without specialized knowledge. While the description might not specifically mention the intended audience, you can assess the overall tone and complexity of the material to determine whether it is accessible to the general public."
    )
    
    # 4. AI topic verification - directly under book node
    ai_topic_verify = evaluator.add_leaf(
        id=f"book_{book_index}_ai_verify",
        desc=f"Book is about AI or closely related topic",
        parent=book_node,
        critical=True  # Changed to critical
    )
    
    if book.title:
        claim = f"The book '{book.title}' is about Artificial Intelligence (AI) or a closely related topic."
    else:
        claim = "Book title is missing"
    
    additional_instruction = """
    To determine if a book is about AI or closely related topics:
    1. Artificial Intelligence (AI) includes machine learning, neural networks, natural language processing, computer vision, robotics, expert systems, etc.
    2. Closely related topics include cognitive science, computational neuroscience, philosophy of mind (as related to AI), data science (when focused on AI applications), and the societal impacts of AI.
    3. Consider the book's description, categories, and reviews to determine its primary focus.
    """
    
    await evaluator.verify(
        claim=claim,
        node=ai_topic_verify,
        sources=book.goodreads_url,
        additional_instruction=additional_instruction
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
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Initialize evaluator ----------------------------- #
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        task_description=TASK_DESCRIPTION,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # -------- 2. Extract book titles first, then details for each book --- #
    # Step 1: Extract just the book titles
    book_titles_extraction = await evaluator.extract(
        prompt=prompt_extract_book_titles(),
        template_class=BookTitles,
        extraction_name="book_titles"
    )

    # Remove duplicates
    book_titles_extraction.titles = list(set(book_titles_extraction.titles))  

    # Extract up to 5 books    
    book_titles = book_titles_extraction.titles[:5]

    # Step 2: Extract detailed information for each book individually
    books = []
    for title in book_titles:
        if title.strip():  # Only process non-empty titles
            book_detail = await evaluator.extract(
                prompt=prompt_extract_book_details(title),
                template_class=BookRecommendation,
                extraction_name=f"book_detail_{title[:30]}"  # Truncate for name
            )
            books.append(book_detail)

    # Pad books to 5
    while len(books) < 5:
        books.append(BookRecommendation(title=None, authors=[], turing_authors=[], goodreads_url=None))
    
    # -------- 3. Build verification tree -------------------------------- #
    # Verify each book directly under root (removed books container)
    for i, book in enumerate(books):
        await verify_single_book(evaluator, root, book, i)
    
    # -------- 4. Get final score and summary ---------------------------- #
    return evaluator.get_summary()