import asyncio
import logging
from typing import Optional, List, Dict, Any

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "rl_course_book"
TASK_DESCRIPTION = """
Search for reinforcement learning (RL) courses offered by Stanford, CMU, and UC Berkeley. For each university, identify one RL course and provide its instructor name(s). From each course's page, select one textbook or reference book that is used or recommended in the course. Ensure that the selected book is different for each university (i.e., no duplicates across the three courses), and that it is available in either paperback or hardcover format from a bookseller, such as Amazon. For each selected book, include its title, author(s), and a link to purchase it.
"""

# Expected universities with normalized names and possible variations
UNIVERSITY_MAP = {
    "stanford": "Stanford",
    "cmu": "CMU",
    "carnegie mellon": "CMU",
    "uc berkeley": "UC Berkeley",
    "berkeley": "UC Berkeley",
    "ucb": "UC Berkeley",
    "university of california berkeley": "UC Berkeley",
    "university of california, berkeley": "UC Berkeley"
}

# List of canonical university names
UNIVERSITIES = ["Stanford", "CMU", "UC Berkeley"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Book(BaseModel):
    title: Optional[str] = None
    authors: Optional[List[str]] = Field(default_factory=list)
    book_urls: Optional[List[str]] = Field(default_factory=list)  # Multiple URLs for book info
    purchase_link: Optional[str] = None


class Course(BaseModel):
    university: Optional[str] = None
    course_name: Optional[str] = None
    instructors: Optional[List[str]] = Field(default_factory=list)
    course_urls: Optional[List[str]] = Field(default_factory=list)  # Multiple URLs for course info
    book: Optional[Book] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                         #
# --------------------------------------------------------------------------- #
def prompt_extract_course_for_university(university: str) -> str:
    """Generate extraction prompt for a specific university"""
    return f"""
    Extract information about the reinforcement learning (RL) course from {university}.
    
    Look for any mention of {university} (or variations like '{university} University', 'Carnegie Mellon', 'UC Berkeley', 'UCB', etc.) 
    and extract the following information:
    
    1. The university name (normalize to "{university}")
    2. The RL course name
    3. The instructor name(s) for that course
    4. ALL course-related URLs or webpage links (course_urls as a list)
    5. Information about a recommended book for the course, including:
       - The book title
       - The book author(s)
       - ALL URLs where book information can be found (book_urls as a list)
       - A link to purchase the book (purchase_link)

    IMPORTANT: 
    - Extract ALL relevant URLs, not just one. Course information might be spread across multiple pages.
    - If the answer doesn't mention {university} or doesn't provide information about an RL course from {university}, 
      return null values for all fields.
    - Only extract information that is specifically about {university}'s RL course, not other universities.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_university_name(univ_name: str) :
    """
    Normalize university name to a canonical form.
    Returns the canonical name if found, otherwise None.
    """
    if not univ_name:
        return None

    # Convert to lowercase for matching
    univ_lower = univ_name.lower()

    # Check if the name or a part of it matches our known universities
    for key, value in UNIVERSITY_MAP.items():
        if key in univ_lower:
            return value

    return None


# --------------------------------------------------------------------------- #
# Book uniqueness verification                                               #
# --------------------------------------------------------------------------- #
async def verify_book_uniqueness(
       evaluator: Evaluator,
       parent_node,
       university_courses: Dict[str, Course],
) -> None:
   """
   Verify that the final recommended books from different universities are completely unique.
   """
   # Collect book titles from the extracted course information
   book_titles = []
   book_authors = []
   for university, course in university_courses.items():
       if course and course.book and course.book.title and course.book.authors:
           book_titles.append(course.book.title)
           book_authors.append(course.book.authors if course.book.authors else [])
   
   # If less than 2 books, automatically pass (no duplicates possible)
   if len(book_titles) <= 1:
       evaluator.add_custom_node(
           result=True,
           id="book_uniqueness_check",
           desc="The final recommended books from different universities are completely unique (no duplicates)",
           parent=parent_node,
           critical=True,
       )
       return

   # Create a verification node for uniqueness
   node = evaluator.add_leaf(
       id="book_uniqueness_check",
       desc="The final recommended books from different universities are completely unique (no duplicates)",
       parent=parent_node,
       critical=True,
   )
   
   # Create claim listing all books and their universities
   book_list = []
   for i, (title, authors) in enumerate(zip(book_titles, book_authors)):
        authors_str = ', '.join(authors)
        book_list.append(f"{i+1}: '{title}' by {authors_str}")
   
   books_description = "\n".join(book_list)
   
   claim = f"""Given the following list of recommended books:

{books_description}

All the book titles in this list completely unique (i.e., no book appears multiple times in this list)."""
   
   additional_instruction = "Compare the book titles carefully. Consider variations in spelling, subtitles, or editions as the same book. For example, 'Reinforcement Learning: An Introduction' and 'Reinforcement Learning: An Introduction (2nd Edition)' should be considered the same book."
   
   await evaluator.verify(
       claim=claim,
       node=node,
       sources=None,
       additional_instruction=additional_instruction,
   )


# --------------------------------------------------------------------------- #
# Course verification functions                                               #
# --------------------------------------------------------------------------- #
async def verify_course_exists(
        evaluator: Evaluator,
        parent_node,
        university: str,
        course: Course,
) -> None:
    """
    Verify that the RL course exists at the specified university.
    """
    # Check if course URLs exist
    course_exists_check = evaluator.add_custom_node(
        result=bool(course and course.course_urls and len(course.course_urls) > 0),
        id=f"{university.lower().replace(' ', '_')}_course_urls_exist",
        desc=f"Course URLs are provided for {university}",
        parent=parent_node,
        critical=True,
    )

    # Verify the URLs point to valid RL course pages at the university
    if course.course_name:
        claim = f"This webpage is for a reinforcement learning (RL) course at {university} with course name '{course.course_name}'."
    else:
        claim = f"This webpage is for a reinforcement learning (RL) course at {university}."

    node = evaluator.add_leaf(
        id=f"{university.lower().replace(' ', '_')}_course_exists",
        desc=f"An RL course exists at {university} and can be accessed via the provided URLs",
        parent=parent_node,
        critical=True,
    )

    await evaluator.verify(
        claim=claim,
        node=node,
        sources=course.course_urls if course else [],
    )


async def verify_book_recommended(
        evaluator: Evaluator,
        parent_node,
        university: str,
        course: Course,
) -> None:
    """
    Verify that the book is recommended in the RL course.
    """
    book_parent_node = evaluator.add_parallel(
        id=f"{university.lower().replace(' ', '_')}_book",
        desc=f"Verify the book is recommended by the course for {university}",
        parent=parent_node,
        critical=False
    )

    # Check if book title exists and course URLs exist
    book_exists_check = evaluator.add_custom_node(
        result=bool(course and course.book and course.book.title and course.course_urls),
        id=f"{university.lower().replace(' ', '_')}_book_info_exists",
        desc=f"Book title and course URLs are provided for {university}",
        parent=book_parent_node,
        critical=True,
    )

    node = evaluator.add_leaf(
        id=f"{university.lower().replace(' ', '_')}_book_recommended",
        desc=f"The book is recommended or used as reference in the {university} RL course",
        parent=book_parent_node,
        critical=True,
    )

    # Verify the book is mentioned/recommended on the course pages
    claim = f"This page is related to a course. And the book titled '{course.book.title if course and course.book else 'N/A'}' is mentioned, recommended, or used as a reference in this course."
    
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=course.course_urls if course else [],
    )


async def verify_instructors_correct(
        evaluator: Evaluator,
        parent_node,
        university: str,
        course: Course,
) -> None:
    """
    Verify that the instructors are correct for the RL course.
    """
    instructor_parent_node = evaluator.add_parallel(
        id=f"{university.lower().replace(' ', '_')}_instructors",
        desc=f"Verify instructors of the course for {university}",
        parent_node=parent_node,
        critical=False  # Non-critical: instructor error shouldn't fail entire university
    )
    # Check if instructors are listed and course URLs exist
    instructors_exist_check = evaluator.add_custom_node(
        result=bool(course and course.instructors and len(course.instructors) > 0 and course.course_urls),
        id=f"{university.lower().replace(' ', '_')}_instructors_exist",
        desc=f"Instructors and course URLs are provided for {university}",
        parent=instructor_parent_node,
        critical=True,
    )

    node = evaluator.add_leaf(
        id=f"{university.lower().replace(' ', '_')}_instructors_correct",
        desc=f"The instructors listed for the {university} RL course are correct",
        parent=instructor_parent_node,
        critical=True,
    )

    # Create a claim about the instructors
    instructors_str = ", ".join(course.instructors) if course and course.instructors else "N/A"
    claim = f"This is a webpage for a course, and the instructor(s) for this course include {instructors_str}."

    # Verify the instructors against the course pages
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=course.course_urls if course else [],
    )


# --------------------------------------------------------------------------- #
# Book verification functions                                                 #
# --------------------------------------------------------------------------- #
async def verify_book_authors_correct(
        evaluator: Evaluator,
        parent_node,
        university: str,
        course: Course,
) -> None:
    """
    Verify that the book authors are correct.
    """
    # Check if book authors exist and book URLs exist
    authors_exist_check = evaluator.add_custom_node(
        result=bool(course and course.book and course.book.authors and 
                   len(course.book.authors) > 0 and course.book.book_urls),
        id=f"{university.lower().replace(' ', '_')}_book_authors_exist",
        desc=f"Book authors and book URLs are provided for {university}",
        parent=parent_node,
        critical=True,
    )

    node = evaluator.add_leaf(
        id=f"{university.lower().replace(' ', '_')}_book_authors_correct",
        desc=f"The authors listed for the {university} RL course book are correct",
        parent=parent_node,
        critical=True,
    )

    # Create a claim about the book authors
    book_title = course.book.title if course and course.book else "N/A"
    authors_str = ", ".join(course.book.authors) if course and course.book and course.book.authors else "N/A"
    claim = f"The book titled '{book_title}' is written by {authors_str}."

    # Verify the authors against book information pages
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=course.book.book_urls if course and course.book else [],
    )


async def verify_purchase_link_valid(
        evaluator: Evaluator,
        parent_node,
        university: str,
        course: Course,
) -> None:
    """
    Verify that the book has a valid purchase link.
    """
    # Check if purchase link exists
    purchase_link_exists = evaluator.add_custom_node(
        result=bool(course and course.book and course.book.purchase_link),
        id=f"{university.lower().replace(' ', '_')}_purchase_link_exists",
        desc=f"Purchase link is provided for {university} book",
        parent=parent_node,
        critical=True,
    )

    node = evaluator.add_leaf(
        id=f"{university.lower().replace(' ', '_')}_purchase_link_valid",
        desc=f"The book for the {university} RL course has a valid purchase link",
        parent=parent_node,
        critical=True,
    )

    # Verify purchase link validity
    book_title = course.book.title if course and course.book else "N/A"
    claim = f"This webpage is for purchasing the book titled '{book_title}'. "
    
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=course.book.purchase_link if course and course.book else None,
    )


async def verify_book_format_correct(
        evaluator: Evaluator,
        parent_node,
        university: str,
        course: Course,
) -> None:
    """
    Verify that the book is available in paperback or hardcover format.
    """
    # Check if purchase link exists
    purchase_link_exists = evaluator.add_custom_node(
        result=bool(course and course.book and course.book.purchase_link),
        id=f"{university.lower().replace(' ', '_')}_purchase_link_exists_format",
        desc=f"Purchase link exists for format verification for {university}",
        parent=parent_node,
        critical=True,
    )

    node = evaluator.add_leaf(
        id=f"{university.lower().replace(' ', '_')}_book_format_correct",
        desc=f"The book for the {university} RL course is available in paperback or hardcover format",
        parent=parent_node,
        critical=True,
    )

    # Verify book format (paperback or hardcover)
    book_title = course.book.title if course and course.book else "N/A"
    claim = f"The book titled '{book_title}' is listed in paperback or hardcover format on this webpage (no need to be currently available)."
    additional_instruction = "Pay attention to format information from both text and images. Look for terms like 'paperback', 'hardcover', 'hardback' from the webpage text, or visual indicators of book format from the screenshot. Don't be too strict on exact wording"

    await evaluator.verify(
        claim=claim,
        node=node,
        sources=course.book.purchase_link if course and course.book else None,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Main university verification functions                                      #
# --------------------------------------------------------------------------- #
async def verify_course_requirements(
        evaluator: Evaluator,
        parent_node,
        university: str,
        course: Course,
) -> None:
    """
    Verify course-related requirements for a university.
    """
    course_node = evaluator.add_sequential(
        id=f"{university.lower().replace(' ', '_')}_course_verification",
        desc=f"Course verification for {university} RL course",
        parent=parent_node,
        critical=False,
    )

    # Verify course exists (critical)
    await verify_course_exists(evaluator, course_node, university, course)

    # Verify book recommended (critical)
    await verify_book_recommended(evaluator, course_node, university, course)

    # Verify instructors (non-critical)
    await verify_instructors_correct(evaluator, course_node, university, course)


async def verify_book_requirements(
        evaluator: Evaluator,
        parent_node,
        university: str,
        course: Course,
) -> None:
    """
    Verify book-related requirements for a university.
    """
    book_node = evaluator.add_parallel(
        id=f"{university.lower().replace(' ', '_')}_book_verification",
        desc=f"Book verification for {university} RL course recommended book",
        parent=parent_node,
        critical=False,
    )

    # All book verifications are critical within the book verification node
    await verify_book_authors_correct(evaluator, book_node, university, course)
    await verify_purchase_link_valid(evaluator, book_node, university, course)
    await verify_book_format_correct(evaluator, book_node, university, course)


async def verify_university_requirements(
        evaluator: Evaluator,
        parent_node,
        university: str,
        course: Course,
) -> None:
    """
    Verify all requirements for a specific university in a sequential manner.
    """
    university_node = evaluator.add_sequential(
        id=f"{university.lower().replace(' ', '_')}_verification",
        desc=f"Sequential verification of the RL course and recommended book from {university}",
        parent=parent_node,
        critical=False,  # Non-critical: allows partial scoring across universities
    )

    # STEP 1: Verify course requirements
    await verify_course_requirements(evaluator, university_node, university, course)

    # STEP 2: Verify book requirements
    await verify_book_requirements(evaluator, university_node, university, course)


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
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract course information for each university separately
    university_courses = {}
    
    for university in UNIVERSITIES:
        # Extract course info for this specific university
        course_info = await evaluator.extract(
            prompt=prompt_extract_course_for_university(university),
            template_class=Course,
            extraction_name=f"{university.lower().replace(' ', '_')}_course_info",
        )
        
        # Normalize university name if extracted
        if course_info.university:
            normalized_name = normalize_university_name(course_info.university)
            if normalized_name:
                course_info.university = normalized_name
        
        # Store the course info (even if empty)
        university_courses[university] = course_info

    # Add extraction results as custom info for the summary
    evaluator.add_custom_info(
        {univ: course.dict() for univ, course in university_courses.items()},
        "extracted_courses_by_university"
    )

    # First, verify book uniqueness (critical at root level)
    await verify_book_uniqueness(evaluator, root, university_courses)

    # Then, verify each university's requirements (non-critical at root level)
    for university in UNIVERSITIES:
        course = university_courses[university]
        # Verify this university's requirements
        await verify_university_requirements(evaluator, root, university, course)

    # Return structured result using evaluator's built-in summary
    return evaluator.get_summary()