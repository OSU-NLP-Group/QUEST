import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "univ_president_jan2026"
TASK_DESCRIPTION = (
    "Identify the individual who was appointed to a university president position in January 2026 and meets ALL of the following criteria: "
    "The appointment is to a numbered presidency at the institution (e.g., 15th president, 16th president); "
    "The effective start date is July 1, 2026; "
    "The contract extends through June 30, 2031; "
    "The initial contract length is exactly 5 years; "
    "The base annual salary is at least $2 million; "
    "The individual previously held a chancellor or president position at a different university; "
    "The individual served at their previous institution for at least 10 years; "
    "The appointment is at a public university; "
    "The individual previously served as a law school dean at the same institution they are now joining as president. "
    "Provide the individual's full name and the name of the university where they were appointed as president."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AppointmentInfo(BaseModel):
    """
    Structured extraction of the appointed president information and supporting sources.
    Use strings for flexibility in formats (e.g., 'two million', '5 years', '16th president').
    """
    individual_full_name: Optional[str] = None
    university_name: Optional[str] = None
    position_title: Optional[str] = None  # e.g., "President of XYZ University"
    announcement_date: Optional[str] = None  # e.g., "January 15, 2026"
    presidency_ordinal: Optional[str] = None  # e.g., "16th president"
    effective_start_date: Optional[str] = None  # e.g., "July 1, 2026"
    contract_end_date: Optional[str] = None  # e.g., "June 30, 2031"
    contract_length: Optional[str] = None  # e.g., "5 years"
    base_annual_salary: Optional[str] = None  # e.g., "$2 million"
    previous_role_title: Optional[str] = None  # e.g., "Chancellor" or "President"
    previous_institution_name: Optional[str] = None  # e.g., "University of ABC"
    previous_service_length: Optional[str] = None  # e.g., "10 years"
    is_public_university: Optional[str] = None  # e.g., "public research university"
    prior_law_dean_role: Optional[str] = None  # e.g., "Dean of XYZ Law School at [current university]"
    sources: List[str] = Field(default_factory=list)  # All URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_appointment_info() -> str:
    return """
    Extract the details of the individual appointed as a university president as presented in the answer. 
    Return a single JSON object with the following fields, using strings for all fields:

    - individual_full_name: The full name of the individual appointed as president.
    - university_name: The name of the university where they were appointed president.
    - position_title: The exact position title stated (should indicate 'President' of a university).
    - announcement_date: The date of the appointment announcement, preferably with month and year (e.g., 'January 2026' or 'January 15, 2026').
    - presidency_ordinal: The stated ordinal (e.g., '16th president', '15th president'); if mentioned, extract exactly.
    - effective_start_date: The effective start date of the presidency (e.g., 'July 1, 2026').
    - contract_end_date: The stated contract end date (e.g., 'June 30, 2031').
    - contract_length: The initial contract length phrase (e.g., '5 years', 'five years').
    - base_annual_salary: The stated base annual salary (e.g., '$2 million', '$2,100,000').
    - previous_role_title: The prior role at another university (e.g., 'Chancellor', 'President').
    - previous_institution_name: The name of the prior university for that role.
    - previous_service_length: The duration served at the prior institution (e.g., '10 years', 'more than a decade').
    - is_public_university: Any explicit phrase indicating the appointed institution is public (e.g., 'public university', 'state university', 'public research university').
    - prior_law_dean_role: A phrase indicating the individual previously served as a law school dean at the same institution they are joining as president (e.g., 'Dean of [University] School of Law').
    - sources: An array of all URLs cited in the answer text (include Google Doc/Press release links, official university pages, news articles). If URLs are in markdown, extract the actual URL.

    Rules:
    - Extract only what is explicitly mentioned in the answer text. Do not infer.
    - If a field is not mentioned, set it to null.
    - For 'sources', include all URLs mentioned anywhere in the answer relevant to the appointment.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _safe_str(val: Optional[str]) -> str:
    return val or ""


# --------------------------------------------------------------------------- #
# Main verification construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: AppointmentInfo) -> None:
    """
    Build the verification tree per rubric and run verifications.
    All checks are critical under the main task node to enforce "meets ALL criteria".
    """
    # Critical parent node to aggregate all checks
    task_node = evaluator.add_parallel(
        id="task_verification",
        desc="Identify the educational leader appointed in January 2026 who satisfies all specified criteria and provide both their full name and university name",
        parent=evaluator.root,
        critical=True
    )

    # Existence checks for name and university
    evaluator.add_custom_node(
        result=bool(_safe_str(extracted.individual_full_name).strip()),
        id="individual_name_provided",
        desc="The answer provides the individual's full name",
        parent=task_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(_safe_str(extracted.university_name).strip()),
        id="university_name_provided",
        desc="The answer provides the name of the university where they were appointed as president",
        parent=task_node,
        critical=True
    )

    # Prepare sources for verification (can be empty; framework handles routing)
    sources = extracted.sources

    # 1) Position type: must be a university president position
    node_position = evaluator.add_leaf(
        id="position_type",
        desc="The appointment is to a university president position (not K-12 superintendent, provost, or other role)",
        parent=task_node,
        critical=True
    )
    claim_position = (
        f"The appointment is explicitly to the position of President of {_safe_str(extracted.university_name)}."
    )
    await evaluator.verify(
        claim=claim_position,
        node=node_position,
        sources=sources,
        additional_instruction=(
            "Confirm the new role is 'President' of a university (higher education). "
            "Titles like 'provost', 'superintendent', or 'dean' do not qualify as the appointed role here."
        ),
    )

    # 2) Announcement timing: January 2026
    node_announce = evaluator.add_leaf(
        id="announcement_timing",
        desc="The appointment announcement occurred in January 2026",
        parent=task_node,
        critical=True
    )
    claim_announce = "The appointment announcement occurred in January 2026."
    await evaluator.verify(
        claim=claim_announce,
        node=node_announce,
        sources=sources,
        additional_instruction=(
            "Verify the press release or official announcement date falls within January 2026."
        ),
    )

    # 3) Numbered presidency
    node_numbered = evaluator.add_leaf(
        id="numbered_presidency",
        desc="The position represents a numbered presidency at the institution (e.g., 15th president, 16th president, etc.)",
        parent=task_node,
        critical=True
    )
    ordinal = _safe_str(extracted.presidency_ordinal)
    uni_nm = _safe_str(extracted.university_name)
    claim_numbered = (
        f"The appointment explicitly states that the individual will be the {ordinal} president of {uni_nm}."
        if ordinal.strip()
        else f"The appointment explicitly states a numbered presidency at {uni_nm} (e.g., 15th, 16th)."
    )
    await evaluator.verify(
        claim=claim_numbered,
        node=node_numbered,
        sources=sources,
        additional_instruction=(
            "Look for phrases like 'Xth president' or 'N-th president' explicitly in the sources."
        ),
    )

    # 4) Effective start date: July 1, 2026
    node_start = evaluator.add_leaf(
        id="effective_start_date",
        desc="The effective start date of the presidency is July 1, 2026",
        parent=task_node,
        critical=True
    )
    claim_start = "The effective start date of the presidency is July 1, 2026."
    await evaluator.verify(
        claim=claim_start,
        node=node_start,
        sources=sources,
        additional_instruction="Verify that the sources state a start date of July 1, 2026."
    )

    # 5) Contract end date: June 30, 2031
    node_end = evaluator.add_leaf(
        id="contract_end_date",
        desc="The contract term extends through June 30, 2031",
        parent=task_node,
        critical=True
    )
    claim_end = "The contract term extends through June 30, 2031."
    await evaluator.verify(
        claim=claim_end,
        node=node_end,
        sources=sources,
        additional_instruction="Verify the stated contract end date is June 30, 2031."
    )

    # 6) Contract length exactly 5 years
    node_length = evaluator.add_leaf(
        id="contract_length",
        desc="The initial contract length is exactly 5 years",
        parent=task_node,
        critical=True
    )
    claim_length = "The initial contract length is exactly 5 years."
    await evaluator.verify(
        claim=claim_length,
        node=node_length,
        sources=sources,
        additional_instruction=(
            "Confirm that the initial contract length is explicitly 5 years. "
            "If the term is from July 1, 2026 through June 30, 2031, that is five years."
        )
    )

    # 7) Base annual salary at least $2 million
    node_salary = evaluator.add_leaf(
        id="base_salary",
        desc="The base annual salary is at least $2 million",
        parent=task_node,
        critical=True
    )
    claim_salary = "The base annual salary is at least $2,000,000."
    await evaluator.verify(
        claim=claim_salary,
        node=node_salary,
        sources=sources,
        additional_instruction=(
            "Check the base annual salary in the sources. It must be ≥ $2,000,000. "
            "Exclude bonuses, incentives, or non-base compensation."
        )
    )

    # 8) Previous position: Chancellor or President at another university
    node_prev_pos = evaluator.add_leaf(
        id="previous_position",
        desc="The individual previously held a chancellor or president position at another university",
        parent=task_node,
        critical=True
    )
    prev_title = _safe_str(extracted.previous_role_title)
    prev_inst = _safe_str(extracted.previous_institution_name)
    claim_prev_pos = (
        f"The individual previously held a {prev_title or 'chancellor or president'} position at {prev_inst or 'another university'}."
    )
    await evaluator.verify(
        claim=claim_prev_pos,
        node=node_prev_pos,
        sources=sources,
        additional_instruction=(
            "Confirm the person previously served as a university 'Chancellor' or 'President' at a different institution."
        )
    )

    # 9) Previous tenure length: at least 10 years
    node_prev_years = evaluator.add_leaf(
        id="previous_tenure_length",
        desc="The individual served at their previous institution for at least 10 years",
        parent=task_node,
        critical=True
    )
    claim_prev_years = "The individual served at their previous institution for at least 10 years."
    await evaluator.verify(
        claim=claim_prev_years,
        node=node_prev_years,
        sources=sources,
        additional_instruction=(
            "Verify that the tenure length at the previous institution is ≥ 10 years (allow phrasing like 'more than a decade')."
        )
    )

    # 10) Previous institution is different than the current one
    node_diff_inst = evaluator.add_leaf(
        id="different_institution",
        desc="The previous institution is a different university than the one they are joining",
        parent=task_node,
        critical=True
    )
    claim_diff_inst = (
        f"The previous institution ({prev_inst}) is different from the appointed institution ({uni_nm})."
    )
    await evaluator.verify(
        claim=claim_diff_inst,
        node=node_diff_inst,
        sources=None,  # Logical comparison; no need for web verification
        additional_instruction=(
            "Judge this as a simple comparison between institution names; treat casing/abbreviations robustly."
        )
    )

    # 11) Appointment is at a public university
    node_public = evaluator.add_leaf(
        id="public_university",
        desc="The appointment is at a public university",
        parent=task_node,
        critical=True
    )
    claim_public = f"{uni_nm} is a public university."
    await evaluator.verify(
        claim=claim_public,
        node=node_public,
        sources=sources,
        additional_instruction=(
            "Verify the institution is public (e.g., state university, public research university)."
        )
    )

    # 12) Prior law school dean role at the same institution
    node_dean = evaluator.add_leaf(
        id="prior_dean_role",
        desc="The individual previously served as a law school dean at the institution they are now joining as president",
        parent=task_node,
        critical=True
    )
    claim_dean = (
        f"The individual previously served as the dean of the law school at {uni_nm}."
    )
    await evaluator.verify(
        claim=claim_dean,
        node=node_dean,
        sources=sources,
        additional_instruction=(
            "Confirm explicit language that the person was the law school dean (e.g., 'Dean of [University] School of Law') at the same university they are now joining as president."
        )
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the January 2026 university president appointment task.
    Returns the standard evaluation summary dict from the evaluator.
    """
    # Initialize evaluator with root parallel node
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured appointment information
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_appointment_info(),
        template_class=AppointmentInfo,
        extraction_name="appointment_info",
    )

    # Add a compact view of extracted key fields for downstream inspection
    evaluator.add_custom_info(
        info={
            "individual_full_name": extracted_info.individual_full_name,
            "university_name": extracted_info.university_name,
            "position_title": extracted_info.position_title,
            "announcement_date": extracted_info.announcement_date,
            "presidency_ordinal": extracted_info.presidency_ordinal,
            "effective_start_date": extracted_info.effective_start_date,
            "contract_end_date": extracted_info.contract_end_date,
            "contract_length": extracted_info.contract_length,
            "base_annual_salary": extracted_info.base_annual_salary,
            "previous_role_title": extracted_info.previous_role_title,
            "previous_institution_name": extracted_info.previous_institution_name,
            "previous_service_length": extracted_info.previous_service_length,
            "is_public_university": extracted_info.is_public_university,
            "prior_law_dean_role": extracted_info.prior_law_dean_role,
            "sources_count": len(extracted_info.sources),
        },
        info_type="extraction_summary",
        info_name="extracted_fields_overview"
    )

    # Build the verification tree and run checks
    await build_verification_tree(evaluator, extracted_info)

    # Return structured evaluation summary
    return evaluator.get_summary()