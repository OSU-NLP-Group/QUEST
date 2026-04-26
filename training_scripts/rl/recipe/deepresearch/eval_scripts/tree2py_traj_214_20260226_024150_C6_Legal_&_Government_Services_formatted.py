import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cabinet_depts_20th_century_locality_2026"
TASK_DESCRIPTION = (
    "Identify three federal Cabinet-level executive departments that meet ALL of the following criteria: "
    "(1) The department was established in the 20th century (between January 1, 1900, and December 31, 1999, inclusive); "
    "(2) The department employs more than 50,000 federal workers as of 2026; "
    "(3) The department's headquarters is located within the Washington-Baltimore-Arlington, DC-MD-VA-WV-PA locality pay area for General Schedule employees. "
    "For each of the three qualifying departments you identify, provide the following information with supporting reference URLs to official government sources: "
    "(A) The department's official full name; (B) The exact formation date of the department (month, day, and year); "
    "(C) The documented number of employees as of 2026; "
    "(D) Confirmation that the department's headquarters location qualifies for the Washington-Baltimore-Arlington, DC-MD-VA-WV-PA locality pay area, including the headquarters location; "
    "(E) The name of at least one major organizational sub-unit (bureau, office, or service) within the department, along with a brief description of its primary function or mission. "
    "Provide reference URLs from official government sources (such as OPM.gov, department websites, USA.gov, or other .gov domains) to support each piece of information."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DepartmentInfo(BaseModel):
    # Identity
    official_name: Optional[str] = None
    identity_sources: List[str] = Field(default_factory=list)

    # Formation date
    formation_date: Optional[str] = None  # exact date string as presented (e.g., "March 3, 1966")
    formation_date_sources: List[str] = Field(default_factory=list)

    # Employees
    employee_count: Optional[str] = None  # as presented (e.g., "approximately 90,000" or "90,000")
    employee_count_year: Optional[str] = None  # year context (e.g., "2026" or "FY 2026")
    employee_count_sources: List[str] = Field(default_factory=list)

    # Locality pay / HQ
    headquarters_location: Optional[str] = None
    locality_sources: List[str] = Field(default_factory=list)

    # Organizational sub-unit
    bureau_name: Optional[str] = None
    bureau_function: Optional[str] = None
    bureau_sources: List[str] = Field(default_factory=list)


class DepartmentsExtraction(BaseModel):
    departments: List[DepartmentInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_departments() -> str:
    return (
        "Extract up to the first three Cabinet-level executive departments mentioned in the answer text that the answer attempts to qualify. "
        "For each department, extract the following fields exactly as presented in the answer:\n"
        "1) official_name: The department's official full name.\n"
        "2) identity_sources: A list of reference URLs that the answer cites to verify the official name (prefer .gov if present). "
        "Only include actual URLs explicitly in the answer; do not invent URLs.\n"
        "3) formation_date: The exact formation date string (month, day, year) as presented.\n"
        "4) formation_date_sources: A list of reference URLs cited to verify the formation date (prefer .gov if present).\n"
        "5) employee_count: The number of employees as of 2026 (string as presented; ranges or approximate words allowed).\n"
        "6) employee_count_year: The year context associated with the employee_count, if provided (e.g., '2026', 'FY 2026'). "
        "If the answer does not clearly state the year, return null.\n"
        "7) employee_count_sources: A list of reference URLs cited to verify the employee count (prefer .gov if present).\n"
        "8) headquarters_location: The department headquarters location string as presented.\n"
        "9) locality_sources: A list of reference URLs cited to verify the headquarters location and/or locality pay area qualification (prefer OPM.gov or other .gov).\n"
        "10) bureau_name: The name of one major organizational sub-unit (bureau, office, or service) within the department.\n"
        "11) bureau_function: A brief description of that sub-unit's primary function or mission as presented.\n"
        "12) bureau_sources: A list of reference URLs cited to verify the sub-unit and/or its mission (prefer .gov if present).\n\n"
        "General rules:\n"
        "- Extract only information explicitly mentioned in the answer; do not add or infer anything.\n"
        "- For any field not present, set it to null (or empty list for URLs).\n"
        "- For URLs, include only valid URLs that appear in the answer (plain or markdown link formats). "
        "If a URL is missing protocol, prepend http://.\n"
        "- If the answer mentions more than three departments, include only the first three.\n"
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_department(
    evaluator: Evaluator,
    parent_node,
    dept: DepartmentInfo,
    index: int,
) -> None:
    """
    Build and verify the rubric sub-tree for a single department.
    """
    # Department node (parallel aggregation, non-critical as per rubric)
    dept_node = evaluator.add_parallel(
        id=f"department_{index + 1}",
        desc=f"{['First','Second','Third'][index]} qualifying department identified and verified",
        parent=parent_node,
        critical=False,
    )

    # ----------------------- Identity ------------------------------------- #
    # Existence: official name provided (critical)
    evaluator.add_custom_node(
        result=bool(dept.official_name) and bool(dept.official_name.strip()) if dept.official_name else False,
        id=f"dept{index + 1}_identity",
        desc="Official full name of the department provided",
        parent=dept_node,
        critical=True,
    )

    # Verify official name via cited sources (critical)
    identity_url_leaf = evaluator.add_leaf(
        id=f"dept{index + 1}_identity_url",
        desc="Reference URL provided verifying the department's official name",
        parent=dept_node,
        critical=True,
    )
    name_claim = f"The department's official full name is '{dept.official_name or ''}'."
    await evaluator.verify(
        claim=name_claim,
        node=identity_url_leaf,
        sources=dept.identity_sources if dept.identity_sources else [],
        additional_instruction=(
            "Confirm the official full name on the provided page(s). Prefer .gov sources if present. "
            "Allow minor formatting or punctuation variants, but the page must clearly indicate the official name."
        ),
    )

    # -------------------- Formation date ---------------------------------- #
    formation_node = evaluator.add_parallel(
        id=f"dept{index + 1}_formation_date",
        desc="Formation date criteria verification",
        parent=dept_node,
        critical=True,
    )

    # Exact date provided (critical)
    date_provided_node = evaluator.add_leaf(
        id=f"dept{index + 1}_date_provided",
        desc="Exact formation date (month, day, year) is provided",
        parent=formation_node,
        critical=True,
    )
    date_provided_claim = (
        f"The provided formation date string '{dept.formation_date or ''}' represents a full date including month, day, and year."
    )
    await evaluator.verify(
        claim=date_provided_claim,
        node=date_provided_node,
        additional_instruction=(
            "Judge whether the string appears to be a complete date (month, day, year). "
            "Examples: 'March 3, 1966', '06/12/1953', or '1953-06-12' count as full dates."
        ),
    )

    # Date lies in 20th century (critical)
    date_range_node = evaluator.add_leaf(
        id=f"dept{index + 1}_date_in_20th_century",
        desc="Formation date falls between January 1, 1900 and December 31, 1999 (inclusive)",
        parent=formation_node,
        critical=True,
    )
    date_range_claim = (
        f"The formation date '{dept.formation_date or ''}' falls between January 1, 1900 and December 31, 1999, inclusive."
    )
    await evaluator.verify(
        claim=date_range_claim,
        node=date_range_node,
        additional_instruction=(
            "Focus solely on whether the given date string is within the stated range. "
            "If the date cannot be interpreted, consider this incorrect."
        ),
    )

    # Verify formation date via sources (critical)
    date_url_leaf = evaluator.add_leaf(
        id=f"dept{index + 1}_date_url",
        desc="Reference URL provided verifying the formation date",
        parent=formation_node,
        critical=True,
    )
    date_claim = f"The department was formed on {dept.formation_date or ''}."
    await evaluator.verify(
        claim=date_claim,
        node=date_url_leaf,
        sources=dept.formation_date_sources if dept.formation_date_sources else [],
        additional_instruction=(
            "Confirm the formation/establishment date on the provided official page(s). Prefer .gov sources."
        ),
    )

    # -------------------- Employee count ---------------------------------- #
    emp_node = evaluator.add_parallel(
        id=f"dept{index + 1}_employee_count",
        desc="Employee count criteria verification",
        parent=dept_node,
        critical=True,
    )

    # As-of-2026 count provided (critical)
    # Implemented as a custom existence/condition check: count present AND year mentions 2026
    evaluator.add_custom_node(
        result=(
            bool(dept.employee_count) and (
                isinstance(dept.employee_count_year, str) and ("2026" in dept.employee_count_year)
            )
        ),
        id=f"dept{index + 1}_count_provided",
        desc="Employee count as of 2026 is provided",
        parent=emp_node,
        critical=True,
    )

    # Count exceeds 50,000 (critical, logical check)
    count_threshold_leaf = evaluator.add_leaf(
        id=f"dept{index + 1}_count_exceeds_threshold",
        desc="Employee count exceeds 50,000",
        parent=emp_node,
        critical=True,
    )
    threshold_claim = (
        f"The stated employee count '{dept.employee_count or ''}' is greater than 50,000."
    )
    await evaluator.verify(
        claim=threshold_claim,
        node=count_threshold_leaf,
        additional_instruction=(
            "Assess only the magnitude implied by the string (e.g., '~90,000' or 'over 100,000' should count as > 50,000). "
            "Do not rely on external sources here."
        ),
    )

    # Verify count via sources (critical)
    count_url_leaf = evaluator.add_leaf(
        id=f"dept{index + 1}_count_url",
        desc="Reference URL provided verifying the employee count",
        parent=emp_node,
        critical=True,
    )
    count_claim = (
        f"The department employs {dept.employee_count or ''} federal workers as of 2026."
    )
    await evaluator.verify(
        claim=count_claim,
        node=count_url_leaf,
        sources=dept.employee_count_sources if dept.employee_count_sources else [],
        additional_instruction=(
            "Confirm that the provided page(s) explicitly reference the employee count and the 2026 context. Prefer .gov sources."
        ),
    )

    # -------------------- Locality pay qualification ---------------------- #
    loc_node = evaluator.add_parallel(
        id=f"dept{index + 1}_locality_pay",
        desc="Locality pay area qualification verification",
        parent=dept_node,
        critical=True,
    )

    # HQ location specified (critical existence)
    evaluator.add_custom_node(
        result=bool(dept.headquarters_location) and bool(dept.headquarters_location.strip()) if dept.headquarters_location else False,
        id=f"dept{index + 1}_headquarters_location",
        desc="Headquarters location is specified",
        parent=loc_node,
        critical=True,
    )

    # HQ qualifies for Washington-Baltimore-Arlington DC locality area (critical)
    qualifies_leaf = evaluator.add_leaf(
        id=f"dept{index + 1}_qualifies_for_dcb",
        desc="Headquarters location qualifies for Washington-Baltimore-Arlington, DC-MD-VA-WV-PA locality pay area",
        parent=loc_node,
        critical=True,
    )
    qualifies_claim = (
        f"The headquarters location '{dept.headquarters_location or ''}' is within the Washington-Baltimore-Arlington, DC-MD-VA-WV-PA General Schedule locality pay area."
    )
    await evaluator.verify(
        claim=qualifies_claim,
        node=qualifies_leaf,
        sources=dept.locality_sources if dept.locality_sources else [],
        additional_instruction=(
            "Use OPM locality pay area definitions if provided, or other official sources. "
            "Verify that the HQ city/county is explicitly included in the Washington-Baltimore-Arlington, DC-MD-VA-WV-PA locality area."
        ),
    )

    # Additional explicit HQ location support (critical)
    loc_url_leaf = evaluator.add_leaf(
        id=f"dept{index + 1}_locality_url",
        desc="Reference URL provided verifying headquarters location or locality pay qualification",
        parent=loc_node,
        critical=True,
    )
    loc_claim = f"The department's headquarters location is '{dept.headquarters_location or ''}'."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_url_leaf,
        sources=dept.locality_sources if dept.locality_sources else [],
        additional_instruction=(
            "Confirm the HQ location from the provided official page(s). Prefer .gov sources."
        ),
    )

    # -------------------- Organizational sub-unit (non-critical) ---------- #
    org_node = evaluator.add_parallel(
        id=f"dept{index + 1}_organizational_unit",
        desc="Major organizational sub-unit identification",
        parent=dept_node,
        critical=False,
    )

    # Sub-unit name provided (critical under org node)
    evaluator.add_custom_node(
        result=bool(dept.bureau_name) and bool(dept.bureau_name.strip()) if dept.bureau_name else False,
        id=f"dept{index + 1}_bureau_name",
        desc="Name of a major bureau, office, or service within the department is provided",
        parent=org_node,
        critical=True,
    )

    # Sub-unit function provided (critical under org node)
    evaluator.add_custom_node(
        result=bool(dept.bureau_function) and bool(dept.bureau_function.strip()) if dept.bureau_function else False,
        id=f"dept{index + 1}_bureau_function",
        desc="Primary function or mission of the organizational sub-unit is described",
        parent=org_node,
        critical=True,
    )

    # Verify sub-unit via sources (critical under org node)
    bureau_url_leaf = evaluator.add_leaf(
        id=f"dept{index + 1}_bureau_url",
        desc="Reference URL provided verifying the organizational sub-unit",
        parent=org_node,
        critical=True,
    )
    bureau_claim = (
        f"'{dept.bureau_name or ''}' is an organizational sub-unit within the department and its mission/function is '{dept.bureau_function or ''}'."
    )
    await evaluator.verify(
        claim=bureau_claim,
        node=bureau_url_leaf,
        sources=dept.bureau_sources if dept.bureau_sources else [],
        additional_instruction=(
            "Verify both the existence of the sub-unit within the department and a brief description of its mission or primary function. Prefer .gov sources."
        ),
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
    Evaluate an answer for the Cabinet-level departments criteria task.
    """
    # Initialize evaluator (root is non-critical parallel aggregator to avoid critical-child constraint)
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
        default_model=model,
    )

    # Extract departments info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_departments(),
        template_class=DepartmentsExtraction,
        extraction_name="departments_extraction",
    )

    # Select up to 3 departments; pad with empty entries if fewer
    departments: List[DepartmentInfo] = list(extracted.departments[:3])
    while len(departments) < 3:
        departments.append(DepartmentInfo())

    # Optional: record a custom summary of extracted names
    evaluator.add_custom_info(
        info={"selected_departments": [d.official_name for d in departments]},
        info_type="extraction_summary",
        info_name="selected_departments_summary",
    )

    # Build verification tree for each department
    for i, dept in enumerate(departments):
        await verify_department(evaluator, root, dept, i)

    # Return result summary
    return evaluator.get_summary()