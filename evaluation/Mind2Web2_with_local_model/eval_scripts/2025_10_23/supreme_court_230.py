import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "supreme_court_230"
TASK_DESCRIPTION = """
Locate 5 Supreme Court cases decided since January 1, 2022 that specifically cite Section 230 of the Communications Decency Act. For each case, provide the link to its official opinion PDF file from supremecourt.gov
"""


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class SupremeCourtCase(BaseModel):
    case_name: Optional[str] = None
    decision_date: Optional[str] = None
    pdf_link: Optional[str] = None


class ExtractedCases(BaseModel):
    cases: List[SupremeCourtCase] = Field(default_factory=list)


# Prompt for extracting case information
def prompt_extract_cases() -> str:
    return """
    Extract information about Supreme Court cases mentioned in the answer. For each case, extract:

    1. case_name: The full name of the Supreme Court case
    2. decision_date: The date when the case was decided (if mentioned)
    3. pdf_link: The URL to the official opinion PDF file from supremecourt.gov

    If any field is not explicitly provided in the answer, set it to null.
    Extract all cases mentioned in the answer, even if there are more than 5.
    """


# --------------------------------------------------------------------------- #
# Verification functions for individual case checks                           #
# --------------------------------------------------------------------------- #
async def verify_case(
    evaluator: Evaluator,
    parent_node,
    case: SupremeCourtCase,
    case_index: int,
) -> None:
    """
    Verify a single Supreme Court case using a sequential verification process.
    """
    # Create a sequential node for this case
    case_name = case.case_name if case.case_name else f"Case {case_index + 1}"
    case_node = evaluator.add_parallel(
        id=f"case_{case_index}",
        desc=f"Case {case_index + 1}: {case_name}",
        parent=parent_node,
        critical=False  # Each case is non-critical for partial scoring
    )

    # Verify PDF link is from supremecourt.gov
    pdf_valid = evaluator.add_custom_node(
        result=bool(case.pdf_link and "supremecourt.gov" in case.pdf_link.lower() and "pdf" in case.pdf_link.lower()),
        id=f"case_{case_index}_pdf_valid",
        desc=f"PDF link is provided and is from supremecourt.gov for case {case_index + 1}",
        parent=case_node,
        critical=True
    )


    # Add verification leaf for name and date
    name_date_leaf = evaluator.add_leaf(
        id=f"case_{case_index}_name_date_check",
        desc=f"Verify case name matches '{case_name}' and was decided after January 1, 2022",
        parent=case_node,
        critical=True
    )

    if case.case_name:
        claim = f"This Supreme Court case is named '{case.case_name}' and was decided on or after January 1, 2022."
    else:
        claim = f"This Supreme Court case was decided on or after January 1, 2022."
    await evaluator.verify(
        claim=claim,
        node=name_date_leaf,
        sources=case.pdf_link,
        additional_instruction="(1) If the case name is provide in the claim, verify the case name matches the case in the document (allow for minor variations in formatting). (2) Veirfy that the case was decided on or after January 1, 2022. Look for the decision date in the document header or conclusion."
    )

    # Add verification leaf for Section 230
    section_leaf = evaluator.add_leaf(
        id=f"case_{case_index}_section_230_check",
        desc=f"Verify case cites Section 230 of the Communications Decency Act",
        parent=case_node,
        critical=True
    )

    await evaluator.verify(
        claim="This Supreme Court case specifically cites Section 230 of the Communications Decency Act.",
        node=section_leaf,
        sources=case.pdf_link,
        additional_instruction="Carefully examine the PDF. Verify that the document specifically mentions or discusses 'Section 230' or '§230' of the Communications Decency Act. The mere mention of the Communications Decency Act without specific reference to Section 230 is insufficient."
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
    # -------- 1. Initialize evaluator ------------------------------------ #
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        agent_name=agent_name,
        answer_name=answer_name,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # -------- 2. Extract cases from the answer -------------------------- #
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_cases(),
        template_class=ExtractedCases,
        extraction_name="supreme_court_cases"
    )

    # -------- 3. Build verification tree -------------------------------- #
    # Pad to ensure we have exactly 5 cases
    cases_to_verify = extracted_info.cases[:5]
    while len(cases_to_verify) < 5:
        cases_to_verify.append(SupremeCourtCase())  # Empty model instance

    # Verify each case
    for i, case in enumerate(cases_to_verify):
        await verify_case(
            evaluator=evaluator,
            parent_node=evaluator.root,
            case=case,
            case_index=i
        )

    # -------- 4. Get final score and summary ---------------------------- #
    final_score = evaluator.score()

    # Add custom information
    evaluator.add_custom_info({
        "total_cases_provided": len(extracted_info.cases),
        "cases_evaluated": 5,
        "expected_cases": 5
    }, "evaluation_stats")

    # -------- 5. Return structured result ------------------------------- #
    return evaluator.get_summary()