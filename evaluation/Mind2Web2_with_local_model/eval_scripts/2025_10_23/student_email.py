import asyncio
import logging
from typing import Optional, List, Dict

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "student_email"
TASK_DESCRIPTION = """
Visit the homepage of Yu Su, a faculty member in the Computer Science and Engineering department at The Ohio State University. Identify his current PhD students as listed on the website. Then, for each student, find and provide their individual OSU email addresses.
"""

YU_SU_HOMEPAGE = "https://ysu1989.github.io/"
YU_SU_STUDENT_PAGE = "https://ysu1989.github.io/#student"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GroundTruthStudents(BaseModel):
    """Model to represent the ground truth student names from Yu Su's homepage."""
    student_names: List[str] = Field(default_factory=list)


class StudentsList(BaseModel):
    """Model to represent a list of student names extracted from the answer."""
    names: List[str] = Field(default_factory=list)


class StudentEmail(BaseModel):
    """Model to represent a student's email and sources."""
    email: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AnswerSources(BaseModel):
    """Model to extract sources for student list identification."""
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_ground_truth() -> str:
    return """
    From the webpage content, extract a list of CURRENT PhD STUDENTS mentioned on Yu Su's homepage.
    Look for a section specifically showing his "Current Students" or "PhD Students" or similar.

    IMPORTANT:
    1. Return the list of full student names as they appear on the webpage.
    2. Only include current PhD students - do NOT include postdoctoral researchers, former students, 
       alumni, visiting students, or any other non-PhD students.
    3. Look for explicit indicators on the page that show these are PhD students, not other types of students or researchers.

    Return the names of all current PhD students only.
    """


def prompt_extract_student_names_from_answer() -> str:
    return """
    Extract ALL student names mentioned in the answer as Yu Su's current PhD students.

    IMPORTANT:
    1. Extract ALL student names that are presented as Yu Su's current PhD students
    2. Return them in the order they appear in the answer
    3. Only extract student names - do not include emails or other information
    4. Return the list of full student names as they appear in the answer

    This is important because we need to evaluate whether the answer contains hallucinated or incorrect student names.
    """


def prompt_extract_answer_sources() -> str:
    return """
    Extract the source URLs mentioned in the answer that are used to identify Yu Su's PhD students.

    They could be:
    1. URLs for Yu Su's homepage or personal website
    2. Links to Ohio State University pages that list Yu Su's students
    3. Department pages or lab websites that show Yu Su's research group members

    Return only the URLs that would help verify the list of Yu Su's PhD students.
    """


def prompt_extract_student_email(student_name: str) -> str:
    return f"""
    Extract the email address and source URLs provided in the answer for the student named '{student_name}'.
    If the answer doesn't mention this specific student, return null for the email.
    If an email is provided but no sources are given, return the email with an empty sources list.

    For the email:
    1. Only extract the exact email address mentioned for this specific student
    2. If no email is provided for this student, return null

    For the sources:
    1. Extract any URLs that are specifically linked to this student's information
    2. If no specific URLs are provided for this student, extract general URLs that might contain email information
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_homepage_source(
        evaluator: Evaluator,
        parent_node,
        sources: List[str],
) -> None:
    """
    Verify that the answer provides Yu Su's homepage as a source.
    This is a critical node - the task specifically requires finding students from his homepage.
    """
    source_parent = evaluator.add_parallel(
        id="homepage_source_verification",
        desc="Verify that the answer includes Yu Su's homepage as a source",
        parent=parent_node,
        critical=True,
    )
    
    # Check if Yu Su's homepage is in the sources
    yu_su_homepage_found = False
    for url in sources:
        if url.startswith(YU_SU_HOMEPAGE) or url == YU_SU_HOMEPAGE or url == YU_SU_STUDENT_PAGE or YU_SU_HOMEPAGE.lower() in url.lower():
            yu_su_homepage_found = True
            break
    
    evaluator.add_custom_node(
        result=yu_su_homepage_found,
        id="homepage_found",
        desc="Check if Yu Su's homepage is cited as a source",
        parent=source_parent,
        critical=True
    )


async def verify_student(
        evaluator: Evaluator,
        parent_node,
        answer_student_name: str,
        ground_truth_students: List[str],
        student_email_info: StudentEmail,
        student_index: int,
) -> None:
    """
    Verify a student's name and email information.
    """
    # Create a sequential node for this student
    student_node = evaluator.add_sequential(
        id=f"answer_student_{student_index}",
        desc=f"Verification for answer student {student_index + 1}: {answer_student_name}",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial scoring
    )
    
    # Step 1: Verify the student name is in ground truth
    name_node = evaluator.add_leaf(
        id=f"name_verification_answer_{student_index}",
        desc=f"Verify that the student name '{answer_student_name}' is among Yu Su's actual PhD students",
        parent=student_node,
        critical=True,  # Critical within the sequential flow
    )
    
    claim = f"The student name '{answer_student_name}' is among Yu Su's current PhD students (ground truth): {', '.join(ground_truth_students)}"
    
    await evaluator.verify(
        claim=claim,
        node=name_node,
        additional_instruction="The verification should be considered passed if the student name from the answer matches any name in the ground truth list, accounting for minor variations in formatting (e.g., with or without middle names/initials, different capitalizations).",
    )
    
    # Step 2: Verify the student's email
    email_verification_parent = evaluator.add_parallel(
        id=f"email_verification_parent_{student_index}",
        desc=f"Email verification for {answer_student_name}",
        parent=student_node,
        critical=False,
    )
    
    # Check if email exists and is valid OSU format
    email_exists_and_valid = (
        student_email_info.email is not None and 
        (student_email_info.email.endswith('@osu.edu') or
         student_email_info.email.endswith('@buckeyemail.osu.edu') or
         '.osu.edu' in student_email_info.email)
    )
    
    evaluator.add_custom_node(
        result=email_exists_and_valid,
        id=f"email_exists_valid_{student_index}",
        desc=f"Check if valid OSU email exists for {answer_student_name}",
        parent=email_verification_parent,
        critical=True
    )
    
    # Verify email against sources
    email_verify_node = evaluator.add_leaf(
        id=f"email_source_verification_{student_index}",
        desc=f"Verify email '{student_email_info.email}' can be confirmed from sources",
        parent=email_verification_parent,
        critical=True,
    )
    
    claim = f"The email address '{student_email_info.email}' belongs to the student '{answer_student_name}' at Ohio State University"
    
    # Use the original, more strict instruction
    instruction = """Consider the verification passed if the source confirms this is the student's OSU email address. The email should be an @osu.edu email address or another official Ohio State University domain.
Note that the student should be a PhD student advised (or co-advised) by Prof. Yu Su, likely affiliated with the Computer Science and Engineering Department. Considering the potential for multiple students sharing the same name at OSU, if the source clearly indicates discrepancies—such as the student belonging to an unrelated department or explicitly listed as an undergraduate—then the verification should be marked as failed.
"""
    
    await evaluator.verify(
        claim=claim,
        node=email_verify_node,
        sources=student_email_info.sources,
        additional_instruction=instruction,
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer to the student email task.

    The evaluation verifies:
    1. That the answer cites Yu Su's homepage as a source
    2. For each student in the answer (up to the number of ground truth students):
       - That the student name is among Yu Su's actual PhD students
       - That the student's email is correct and can be verified from provided sources
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
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

    # -------- 2. First, extract ground truth student names --------------- #
    logger.info("Extracting ground truth student names from Yu Su's homepage")
    ground_truth = await evaluator.extract(
        prompt=prompt_extract_ground_truth(),
        template_class=GroundTruthStudents,
        extraction_name="ground_truth_students",
        source=YU_SU_STUDENT_PAGE,
    )

    ground_truth_students = ground_truth.student_names
    num_ground_truth_students = len(ground_truth_students)
    logger.info(f"Ground truth students ({num_ground_truth_students}): {ground_truth_students}")

    # -------- 3. Extract ALL student names from the answer --------------- #
    logger.info("Extracting ALL student names from the answer")
    answer_students = await evaluator.extract(
        prompt=prompt_extract_student_names_from_answer(),
        template_class=StudentsList,
        extraction_name="answer_students",
    )

    # Filter to only the first K students (where K = number of ground truth students)
    answer_students_filtered = answer_students.names[:num_ground_truth_students]
    logger.info(f"Answer students (all): {answer_students.names}")
    logger.info(f"Answer students (filtered to first {num_ground_truth_students}): {answer_students_filtered}")

    # -------- 4. Extract sources from the answer ------------------------- #
    logger.info("Extracting sources from the answer")
    answer_sources = await evaluator.extract(
        prompt=prompt_extract_answer_sources(),
        template_class=AnswerSources,
        extraction_name="answer_sources",
    )

    # -------- 5. Build verification tree -------------------------------- #
    
    # First, verify that the answer cites Yu Su's homepage
    source_verification = evaluator.add_sequential(
        id="source_verification",
        desc="Verify that the answer cites Yu Su's homepage as a source",
        critical=True,
    )

    await verify_homepage_source(
        evaluator=evaluator,
        parent_node=source_verification,
        sources=answer_sources.sources,
    )

    # Create a node to verify students from the answer
    students_verification = evaluator.add_parallel(
        id="students_verification",
        desc="Verify each student mentioned in the answer (up to the number of ground truth students)",
    )

    # Pad the student list with empty StudentEmail objects if needed
    student_emails = []
    for i, answer_student_name in enumerate(answer_students_filtered):
        logger.info(f"Extracting email information for student: {answer_student_name}")
        student_email_info = await evaluator.extract(
            prompt=prompt_extract_student_email(answer_student_name),
            template_class=StudentEmail,
            extraction_name=f"student_{i}_email",
        )
        student_emails.append(student_email_info)
    
    # Pad with empty objects if fewer students than ground truth
    while len(student_emails) < num_ground_truth_students:
        student_emails.append(StudentEmail())
        answer_students_filtered.append("")  # Empty name for missing students
    
    # Process all students (real and padded)
    for i in range(num_ground_truth_students):
        await verify_student(
            evaluator=evaluator,
            parent_node=students_verification,
            answer_student_name=answer_students_filtered[i] if i < len(answer_students.names) else f"Missing student {i + 1}",
            ground_truth_students=ground_truth_students,
            student_email_info=student_emails[i],
            student_index=i,
        )

    # -------- 6. Add custom info ---------------------------------------- #
    evaluator.add_custom_info({
        "answer_students_all": answer_students.names,
        "answer_students_evaluated": answer_students_filtered[:len(answer_students.names)],
        "extracted_sources": answer_sources.sources,
    }, "extraction_summary")

    # -------- 7. Return structured result ------------------------------- #
    return evaluator.get_summary()