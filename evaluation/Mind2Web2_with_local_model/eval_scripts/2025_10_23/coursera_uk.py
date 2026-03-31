import asyncio
import logging
from typing import Dict, List, Optional

import openai
from mind2web2.evaluator import Evaluator
from mind2web2.utils.cache_filesys import CacheFileSys
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "coursera_uk"
TASK_DESCRIPTION = """
I want to learn computer programming skills on Coursera. Could you please recommend a course related to programming and algorithms (i.e., with skill tag computer programming), as well as a Python-related course (i.e., with skill tag Python programming), both offered by UK universities and with a rating of 4.5 or higher? Please provide the homepage links for these courses on Coursera.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CourseInfo(BaseModel):
    """Model for information about a single course"""
    name: Optional[str] = None
    url: Optional[str] = None


class ExtractedCourses(BaseModel):
    """Model for all courses extracted from the answer"""
    programming_courses: List[CourseInfo] = Field(default_factory=list)
    python_courses: List[CourseInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_courses() -> str:
    return """
    Extract all courses mentioned in the answer. The answer should recommend courses in two categories:
    1. Programming and algorithms courses (with skill tag "computer programming")
    2. Python-related courses (with skill tag "Python programming")

    For each course, extract the following information:
    - name: The name of the course (if mentioned)
    - url: The URL/link to the course on Coursera

    Return all programming/algorithms courses in the 'programming_courses' list and all Python courses in the 'python_courses' list.
    If a course belongs to both categories, include it in both lists.
    If a particular field is not mentioned in the answer, set it to null.
    """


# --------------------------------------------------------------------------- #
# Course verification functions                                               #
# --------------------------------------------------------------------------- #
async def verify_course(
        evaluator: Evaluator,
        courses: List[CourseInfo],
        course_type: str,
        skill_tag: str,
) -> None:
    """
    Verify the first course of the specified type in the answer.
    Uses sequential verification: URL validity -> Course requirements verification

    Args:
        evaluator: The evaluator instance
        courses: List of courses of this type
        course_type: "programming" or "python" for ID generation
        skill_tag: The expected skill tag for verification
    """
    
    # Always create all child nodes upfront, even if we'll skip them

    course_node = evaluator.add_parallel(
        f"{course_type}",
        f"Verify for {course_type} course",
        critical=False,
    )

    has_course_with_url = bool(courses and len(courses) > 0 and courses[0].url)
    existence_node = evaluator.add_custom_node(
        result=has_course_with_url,
        id=f"{course_type}_existence",
        desc=f"Verify whether the basic information for the {course_type} type is provided",
        parent=course_node,
        critical=True
    )

    # Get the first course
    course = courses[0]

    # 1. First check: Valid Coursera URL
    url_node = evaluator.add_leaf(
        f"{course_type}_course_url",
        f"Verify that the URL for the {course_type} course is a valid Coursera course page",
        parent=course_node,
        critical=True,
    )

    claim = f"The URL '{course.url}' is a valid Coursera course page."
    await evaluator.verify(
        claim=claim,
        node=url_node,
        sources=course.url,
    )

    # 2. Second check: All course requirements (only if URL is valid)

    requirements_node = evaluator.add_leaf(
        f"{course_type}_course_requirements",
        f"Verify that the {course_type} course meets all requirements: UK university, rating ≥4.5, and {skill_tag} skill tag",
        parent=course_node,
        critical=True,
        status="initialized"
    )

    # Create a comprehensive claim that covers all requirements
    course_name = f"The course '{course.name}'" if course.name else "This course"
    claim = f"""{course_name} meets all the following requirements:
    1. It is offered by a university in the United Kingdom (UK)
    2. It has a rating of 4.5 or higher
    3. It has the skill tag '{skill_tag}'"""

    await evaluator.verify(
        claim=claim,
        node=requirements_node,
        sources=course.url,
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
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.

    The evaluation checks whether the answer provides:
    1. A course related to programming/algorithms with "computer programming" skill tag
    2. A Python-related course with "Python programming" skill tag

    Both courses must be:
    - Offered by UK universities
    - Have a rating of 4.5 or higher
    - Have valid Coursera homepage links
    - Have the appropriate skill tags

    Only the first course of each type is evaluated. Verification is sequential,
    with URL validity checked first, followed by comprehensive requirements verification.
    """
    # -------- 1. Initialize evaluator ----------------------------------- #
    evaluator = Evaluator()
    
    evaluator.initialize(
        task_id=TASK_ID,
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
    extracted_courses = await evaluator.extract(
        prompt=prompt_extract_courses(),
        template_class=ExtractedCourses,
        extraction_name="extracted_courses"
    )

    # -------- 3. Verify only the first course of each type -------------- #
    await verify_course(
        evaluator,
        extracted_courses.programming_courses,
        "programming",
        "computer programming"
    )
    
    await verify_course(
        evaluator,
        extracted_courses.python_courses,
        "python",
        "Python programming"
    )

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()