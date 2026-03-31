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
TASK_ID = "female_scholarships"
TASK_DESCRIPTION = """
Please compile a list of five scholarships or fellowships that are specifically targeted at, or exclusively available to, female graduate students. These opportunities should be open to female U.S. citizens who are enrolled in computer science graduate programs at US universities. For each opportunity, include the following details: 1) the name of the scholarship or fellowship, 2) the award amount, 3) the application deadline (or the typical application deadline of it if that of current year is unavailable).
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ScholarshipDetails(BaseModel):
    name: Optional[str] = None
    award_amount: Optional[str] = None
    deadline: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ScholarshipInfo(BaseModel):
    scholarships: List[ScholarshipDetails] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_scholarships() -> str:
    return """
    Extract information about scholarships or fellowships mentioned in the answer. For each scholarship, extract:

    1. The name of the scholarship or fellowship
    2. The award amount (this can be a specific amount, a range, or noted as "Varies")
    3. The application deadline (or typical deadline if exact date isn't specified)
    4. Any URLs provided in the answer that are associated with this scholarship

    Return a list of extracted scholarships, with each containing these details. If any piece of information is missing, return null for that field.

    Note: Extract ALL scholarships mentioned in the answer, not just 5, as the answer might contain more than requested.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_scholarship_details(
        evaluator: Evaluator,
        parent_node,
        scholarship: ScholarshipDetails,
        index: int,
) -> None:
    """
    Verify details of a single scholarship. Each scholarship requires:
    1. Name is provided
    2. Award amount is provided
    3. Deadline is provided
    4. Information is correctly substantiated by URLs
    5. Targets female graduate students in CS
    6. Open to US citizens at US universities
    """
    scholarship_name = scholarship.name or 'Unnamed'
    
    # Create main scholarship node
    scholarship_node = evaluator.add_parallel(
        id=f"scholarship_{index}",
        desc=f"Scholarship #{index + 1}: {scholarship_name}",
        parent=parent_node,
        critical=False,
    )

    # Create a completeness check parent
    completeness_node = evaluator.add_custom_node(
        result=bool(scholarship.name) and bool(scholarship.award_amount) and bool(scholarship.deadline) and bool(scholarship.urls),
        id=f"scholarship_{index}_completeness",
        desc=f"Basic information completeness for {scholarship_name}",
        parent=scholarship_node,
        critical=True,
    )

    # Verify female targeting
    female_node = evaluator.add_leaf(
        id=f"scholarship_{index}_female",
        desc=f"The scholarship '{scholarship_name}' is targeted at or exclusively available to female students",
        parent=scholarship_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"The scholarship '{scholarship_name}' is targeted at or exclusively available to female students.",
        node=female_node,
        sources=scholarship.urls,
        additional_instruction="Even if it doesn't explicitly show its target at female, as long as it highlights its target on diversity, or target at underrepresentative people, or it emphasis female in any aspect (for example, it empasizes enrollement of female)"
    )

    # Verify CS graduate students
    cs_grad_node = evaluator.add_leaf(
        id=f"scholarship_{index}_cs_grad",
        desc=f"The scholarship '{scholarship_name}' is for computer science graduate students",
        parent=scholarship_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"The scholarship '{scholarship_name}' is not closed to computer science graduate students.",
        node=cs_grad_node,
        sources=scholarship.urls,
        additional_instruction="No need to be too strict. As long as it doesn't explicitly mention that it is not available to computer science student, lets assume it's open to CS and make it a pass. Or, in other words, if it does not specify any fields, treat it a pass. A special case is AAUW Selected Professions Fellowship Program, for this, always consider it as correct."
    )

    # Verify US citizens at US universities
    us_node = evaluator.add_leaf(
        id=f"scholarship_{index}_us",
        desc=f"The scholarship '{scholarship_name}' is open to US citizens at US universities",
        parent=scholarship_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"The scholarship '{scholarship_name}' is open to US citizens studying at US universities or does not specify citizenship/university requirements, making it potentially available to US citizens at US universities.",
        node=us_node,
        sources=scholarship.urls,
        additional_instruction="If no explicit requirement on the applicant's nationality and affiliation region is mentioned, assume US citizens from US universities can apply."
    )

    # Create details substantiation parent
    details_node = evaluator.add_parallel(
        id=f"scholarship_{index}_details_substantiation",
        desc=f"Details substantiation for {scholarship_name}",
        parent=scholarship_node,
        critical=True,
    )

    # Verify name substantiation
    name_subst_node = evaluator.add_leaf(
        id=f"scholarship_{index}_name_substantiation",
        desc=f"The scholarship name '{scholarship_name}' is substantiated by at least one provided URL",
        parent=details_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"The scholarship or fellowship is named '{scholarship_name}'.",
        node=name_subst_node,
        sources=scholarship.urls,
    )

    # Verify amount substantiation
    amount_subst_node = evaluator.add_leaf(
        id=f"scholarship_{index}_amount_substantiation",
        desc=f"The award amount '{scholarship.award_amount}' is substantiated by at least one provided URL",
        parent=details_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"The award amount for the '{scholarship_name}' scholarship is {scholarship.award_amount}.",
        node=amount_subst_node,
        sources=scholarship.urls,
        additional_instruction="An exact award amount is not required - an amount range or 'Varies' is acceptable if supported by the source. However, if the source provides an exact amount, then the answer should match that exact amount."
    )

    # Verify deadline substantiation
    deadline_subst_node = evaluator.add_leaf(
        id=f"scholarship_{index}_deadline_substantiation",
        desc=f"The application deadline '{scholarship.deadline}' is substantiated by at least one provided URL",
        parent=details_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"The application deadline for the '{scholarship_name}' scholarship is {scholarship.deadline}.",
        node=deadline_subst_node,
        sources=scholarship.urls,
        additional_instruction="A typical or previous year's deadline is acceptable."
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

    This function extracts scholarship information from the answer,
    verifies that each scholarship meets the requirements (female-targeted,
    CS graduate students, US citizens at US universities), and validates
    that all required details (name, amount, deadline) are present and
    substantiated by reliable sources.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract scholarships from the answer
    scholarships_info = await evaluator.extract(
        prompt=prompt_extract_scholarships(),
        template_class=ScholarshipInfo,
        extraction_name="scholarships_extraction",
    )

    # Add custom info about extraction results
    evaluator.add_custom_info({
        "num_scholarships_found": len(scholarships_info.scholarships),
        "requested_scholarships": 5,
    }, "extraction_summary")

    # Check if at least one scholarship is provided
    evaluator.add_custom_node(
        result=bool(scholarships_info.scholarships),
        id="has_scholarships",
        desc="At least one scholarship or fellowship is provided in the answer",
        critical=True
    )

    # Create a parallel node for individual scholarship verifications
    scholarships_node = evaluator.add_parallel(
        id="scholarships",
        desc="Verification of individual scholarships",
        critical=False,
    )

    # Prepare list of scholarships to evaluate, ensuring we have exactly 5 entries
    # Pad with empty ScholarshipDetails objects if fewer than 5
    scholarships_to_evaluate = list(scholarships_info.scholarships[:5])  # Take first 5 if more provided
    while len(scholarships_to_evaluate) < 5:
        scholarships_to_evaluate.append(ScholarshipDetails())

    # Verify each scholarship
    for i, scholarship in enumerate(scholarships_to_evaluate):
        await verify_scholarship_details(
            evaluator=evaluator,
            parent_node=scholarships_node,
            scholarship=scholarship,
            index=i,
        )

    # Get final results
    return evaluator.get_summary()