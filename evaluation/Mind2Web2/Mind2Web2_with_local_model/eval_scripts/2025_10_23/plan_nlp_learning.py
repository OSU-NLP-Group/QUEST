import asyncio
import logging
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "plan_nlp_learning"
TASK_DESCRIPTION = """
Create a learning path for a freshman with no prior knowledge to learn Natural Language Processing.
The learning path should include five Coursera or EdX courses arranged from beginner-level to advanced-level, ensuring that no course is easier than the one before it. For each course, provide the following details: course provider (Coursera or EdX), difficulty level as specified by the course provider, duration, and a link to the course webpage.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CourseNames(BaseModel):
    """Model for extracting just the names of courses."""
    course_names: List[str] = Field(default_factory=list)

class CourseInfo(BaseModel):
    """Model for detailed information about a single course."""
    name: Optional[str] = None
    provider: Optional[str] = None
    difficulty_level: Optional[str] = None
    duration: Optional[str] = None
    url: Optional[str] = None

class ExtractedCourses(BaseModel):
    """Model to store all extracted courses."""
    courses: List[CourseInfo] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_course_names() -> str:
    return """
    Extract ONLY the names of the NLP courses mentioned in the answer.
    
    Return the course names as a list of strings in the order they appear in the answer.
    If the answer contains more than 5 courses, include all of them.
    If a course doesn't have a clear name, extract whatever identifier is used to refer to it.
    """

def prompt_extract_course_detail(course_name: str) -> str:
    return f"""
    Extract detailed information about the course named: "{course_name}"
    
    Specifically, extract:
    1. The exact name of the course (should match or be similar to "{course_name}")
    2. The provider (Coursera or EdX)
    3. The difficulty level as specified by the course provider
    4. The duration of the course
    5. The URL to the course webpage
    
    If any of these details is not explicitly mentioned in the answer, return null for that field.
    """

# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_course(
    evaluator: Evaluator,
    parent_node,
    course: CourseInfo,
    course_index: int,
    prev_course: Optional[CourseInfo] = None,
) -> None:
    """
    Verify all aspects of a single course.
    """
    course_node = evaluator.add_parallel(
        id=f"course_{course_index}",
        desc=f"Course {course_index+1}: {course.name or 'Unnamed course'} meets all requirements.",
        parent=parent_node,
        critical=False,  # Allow partial credit across courses
    )
    
    # Add existence check for all required fields directly to course_node
    evaluator.add_custom_node(
        result=(course.name is not None and 
                course.provider is not None and 
                course.difficulty_level is not None and 
                course.duration is not None and 
                course.url is not None),
        id=f"course_{course_index}_completeness",
        desc=f"Course {course_index+1} has all required fields (name, provider, difficulty level, duration, URL)",
        parent=course_node,
        critical=True
    )
    
    
    # Course details correctness (against URL) - NOW CRITICAL
    correctness_node = evaluator.add_leaf(
        id=f"course_details_correctness_{course_index}",
        desc=f"Course {course_index+1} details (provider, difficulty level, duration) are accurate according to the course webpage.",
        parent=course_node,
        critical=True,  # Changed to critical
    )

    # Verify course details against URL
    details_claim = f"For the course titled '{course.name or 'the course'}', the following details are correct: "
    details_claim += "it is a Coursera or Edx course related to Natural Language Processing, "
    if course.provider:
        details_claim += f"it is provided by {course.provider}, "
    if course.difficulty_level:
        details_claim += f"its difficulty level is {course.difficulty_level}, "
    if course.duration:
        details_claim += f"its duration is {course.duration}, "
    details_claim = details_claim.rstrip(", ") + "."
    
    await evaluator.verify(
        claim=details_claim,
        node=correctness_node,
        sources=course.url,
        additional_instruction="Check if the course details mentioned (course name, provider, difficulty level, duration) match what's shown on the course webpage. Accept slight variations in wording for difficulty level and duration."
    )

    
    # Difficulty progression check (new)
    if course_index == 0:
        # First course automatically passes progression check
        evaluator.add_custom_node(
            result=True,
            id=f"difficulty_progression_{course_index}",
            desc=f"Course {course_index+1} (first course) automatically passes difficulty progression check",
            parent=course_node,
            critical=True
        )
    else:
        # Check progression from previous course
        if prev_course and prev_course.difficulty_level and course.difficulty_level:
            progression_node = evaluator.add_leaf(
                id=f"difficulty_progression_{course_index}",
                desc=f"Course {course_index+1} is not easier than Course {course_index} (previous course)",
                parent=course_node,
                critical=True,
            )
            progression_claim = (
                f"Course {course_index+1} with difficulty level '{course.difficulty_level}' "
                f"is not easier than Course {course_index} with difficulty level '{prev_course.difficulty_level}'. "
                f"The progression from '{prev_course.difficulty_level}' to '{course.difficulty_level}' "
                f"represents either the same difficulty level or an increase in difficulty."
            )
            
            await evaluator.verify(
                claim=progression_claim,
                node=progression_node,
                additional_instruction=(
                    "Evaluate if the second course is at the same level or more difficult than the first course. "
                    "Consider common difficulty progressions: Beginner/Basic/Introductory → Intermediate → Advanced/Expert. "
                    "Courses at the same level are acceptable (e.g., Intermediate → Intermediate). "
                    "A course should NOT be considered easier than the previous one (e.g., Advanced → Beginner is not allowed)."
                )
            )
        else:
            evaluator.add_custom_node(
                result=False,
                id=f"difficulty_progression_{course_index}",
                desc=f"Cannot check the difficulty progression due to the missing of the previous course or the difficulty level information",
                parent=course_node,
                critical=True
            )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer against the NLP learning path task requirements.
    Uses a two-step extraction process: first extract course names, then details.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
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
    
    # Step 1: Extract the course names
    course_names_result = await evaluator.extract(
        prompt=prompt_extract_course_names(),
        template_class=CourseNames,
        extraction_name="course_names",
    )
    
    # Ensure we have course names
    if not course_names_result.course_names:
        logger.warning("No course names were extracted from the answer.")
        # Create an empty list for further processing
        all_courses = []
    else:
        # Step 2: Extract detailed info for each course
        all_courses = []
        for course_name in course_names_result.course_names:
            course_detail = await evaluator.extract(
                prompt=prompt_extract_course_detail(course_name),
                template_class=CourseInfo,
                extraction_name=f"course_detail_{course_name}",
            )
            
            # Ensure the course name is set even if extraction failed
            if not course_detail.name:
                course_detail.name = course_name
                
            all_courses.append(course_detail)
    
    all_courses = all_courses[:5]
    # Pad the list to ensure we have 5 courses (with empty ones if needed)
    while len(all_courses) < 5:
        all_courses.append(CourseInfo(name=None, provider=None, difficulty_level=None, duration=None, url=None))
    
    # Add custom info about the extracted courses
    evaluator.add_custom_info(
        {"total_courses_extracted": len(course_names_result.course_names)},
        "extraction_stats"
    )
    
    # Verify individual courses (directly under root, no course_count node)
    courses_node = evaluator.add_parallel(
        id="courses_verification",
        desc="Verification of all provided courses.",
        parent=root,
        critical=False,  # Allow partial credit across courses
    )
    
    # Verify each course (limited to first 5)
    courses_to_evaluate = all_courses[:5]
    prev_course = None
    for i, course in enumerate(courses_to_evaluate):
        await verify_single_course(evaluator, courses_node, course, i, prev_course)
        # Update prev_course for next iteration
        prev_course = course
    
    # Return structured result using get_summary()
    return evaluator.get_summary()