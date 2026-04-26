import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "district_enrollment_salary_qualifiers"
TASK_DESCRIPTION = """
Among the following five public school districts—Worcester Public Schools (Massachusetts), Pittsburgh Public Schools (Pennsylvania), Plano Independent School District (Texas), Newark Public Schools (New Jersey), and Arlington Independent School District (Texas)—which districts have both a current student enrollment exceeding 40,000 students AND a superintendent with an annual salary of at least $300,000? Provide the district names along with their current enrollment figures and superintendent salary data with source references.
"""

CANONICAL_DISTRICTS = [
    {"key": "worcester", "name": "Worcester Public Schools", "state": "Massachusetts"},
    {"key": "pittsburgh", "name": "Pittsburgh Public Schools", "state": "Pennsylvania"},
    {"key": "plano_isd", "name": "Plano Independent School District", "state": "Texas"},
    {"key": "newark", "name": "Newark Public Schools", "state": "New Jersey"},
    {"key": "arlington_isd", "name": "Arlington Independent School District", "state": "Texas"},
]

ALLOWED_SALARY_YEARS_TEXT = "2023–24, 2024, or 2025"


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class DistrictData(BaseModel):
    district_name: Optional[str] = None  # Use canonical name exactly as provided
    state: Optional[str] = None
    enrollment: Optional[str] = None                  # e.g., "41,250" or "about 42,000"
    enrollment_year_label: Optional[str] = None       # e.g., "2025–26" or "2024-25" or "current"
    enrollment_urls: List[str] = Field(default_factory=list)

    superintendent_name: Optional[str] = None
    salary: Optional[str] = None                      # e.g., "$325,000"
    salary_year_label: Optional[str] = None           # e.g., "2024-25", "2025"
    salary_urls: List[str] = Field(default_factory=list)


class DistrictsExtraction(BaseModel):
    qualifying_districts: List[str] = Field(default_factory=list)  # Canonical names from the five
    districts: List[DistrictData] = Field(default_factory=list)    # One object per each of the five districts


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_districts() -> str:
    canonical_list = "\n".join(
        [f"- {d['name']} ({d['state']})" for d in CANONICAL_DISTRICTS]
    )
    return f"""
You will extract structured information from the answer about the following EXACT FIVE school districts (use EXACT canonical names):
{canonical_list}

Your tasks:
1) Extract which districts the answer claims meet BOTH conditions:
   – current student enrollment (2025–26 or most recent available) exceeds 40,000
   – superintendent annual salary is at least $300,000 with a salary-year of 2023–24, 2024, or 2025
   Put these canonical district names in 'qualifying_districts'. Only include names from the list above.

2) For EACH of the five districts (even if the answer doesn't list them as qualifying), extract the following (use null for any missing field):
   - district_name: The EXACT canonical district name from the list above.
   - state: The state from the canonical list above.
   - enrollment: The current enrollment figure the answer provides for the district (as text, keep formatting, do NOT normalize; if the answer uses a range/approximation, keep it as-is).
   - enrollment_year_label: The year label attached to the enrollment if stated (e.g., "2025–26", "2024-25", "current", "most recent", etc.).
   - enrollment_urls: All URLs provided in the answer that support the enrollment figure.
   - superintendent_name: The superintendent's name if provided.
   - salary: The superintendent's annual salary figure as provided (as text, keep formatting).
   - salary_year_label: The year label for the salary (e.g., "2023–24", "2024", "2025", "FY 2025", etc.), if provided.
   - salary_urls: All URLs provided that support the salary figure/year.

Rules:
- Only use URLs explicitly present in the answer. Do not fabricate URLs.
- Do not add or infer any numbers that are not explicitly present in the answer.
- Return exactly 5 district objects in 'districts', one per canonical district name, matching the canonical names exactly, in any order.
- If the answer lists districts as qualifying, ensure 'qualifying_districts' only contains canonical names from the five above.
"""


# --------------------------------------------------------------------------- #
# Helper Utilities                                                            #
# --------------------------------------------------------------------------- #
def canonical_names() -> List[str]:
    return [d["name"] for d in CANONICAL_DISTRICTS]


def district_label(name: str, state: Optional[str]) -> str:
    return f"{name} ({state})" if state else name


def find_record_for_canonical(extraction: DistrictsExtraction, canonical_name: str) -> Optional[DistrictData]:
    for rec in (extraction.districts or []):
        if rec and rec.district_name and rec.district_name.strip().lower() == canonical_name.strip().lower():
            return rec
    return None


def has_digits(s: Optional[str]) -> bool:
    if not s:
        return False
    return bool(re.search(r"\d", s))


def urls_nonempty(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


# --------------------------------------------------------------------------- #
# Verification Subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_selection_subset_of_five(evaluator: Evaluator, parent_node, qualifying: List[str]) -> None:
    # Critical leaf: All listed qualifying districts are from the set of the five (no outside districts)
    leaf = evaluator.add_leaf(
        id="selection_subset_of_five",
        desc="All listed qualifying districts are within the specified five only",
        parent=parent_node,
        critical=True,
    )
    allowed = ", ".join(canonical_names())
    listed = ", ".join(qualifying) if qualifying else "(none)"
    claim = (
        "Evaluate whether the answer's list of qualifying districts is a subset of the following five districts only: "
        f"{allowed}. The answer's listed qualifying districts are: {listed}. "
        "Pass if and only if every listed district belongs to the allowed set and no other districts are included."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        additional_instruction="Allow minor name variants (e.g., punctuation, 'ISD' vs 'Independent School District') to count as the same district."
    )


async def verify_listed_district_meets_both(evaluator: Evaluator, parent_node, d: DistrictData) -> None:
    """
    Under Node 1 (selection exactness), confirm that each listed qualifying district truly meets BOTH constraints
    using the answer's cited sources.
    """
    label = district_label(d.district_name or "Unknown District", d.state)

    node = evaluator.add_sequential(
        id=f"qualifying_check_{(d.district_name or 'unknown').lower().replace(' ', '_')}",
        desc=f"Verify both constraints are satisfied for {label} using cited sources",
        parent=parent_node,
        critical=True,
    )

    # Enrollment > 40,000 supported by enrollment sources
    enroll_over_leaf = evaluator.add_leaf(
        id=f"{(d.district_name or 'unknown').lower().replace(' ', '_')}_enroll_gt_40k",
        desc=f"{label}: Enrollment > 40,000 is supported by the cited enrollment source(s)",
        parent=node,
        critical=True,
    )
    claim_enroll = (
        f"The current student enrollment for {label} exceeds 40,000 students (for 2025–26 or the most recent available year)."
    )
    await evaluator.verify(
        claim=claim_enroll,
        node=enroll_over_leaf,
        sources=d.enrollment_urls,
        additional_instruction="Accept small variations or rounding; verify that the cited source clearly indicates enrollment > 40,000."
    )

    # Salary >= $300,000 supported by salary sources
    salary_ge_leaf = evaluator.add_leaf(
        id=f"{(d.district_name or 'unknown').lower().replace(' ', '_')}_salary_ge_300k",
        desc=f"{label}: Superintendent salary ≥ $300,000 is supported by the cited salary source(s)",
        parent=node,
        critical=True,
    )
    claim_salary_ge = (
        f"The superintendent's annual salary for {label} is at least $300,000."
    )
    await evaluator.verify(
        claim=claim_salary_ge,
        node=salary_ge_leaf,
        sources=d.salary_urls,
        additional_instruction="Verify from the cited source(s) that the salary is ≥ $300,000; accept minor formatting differences (e.g., commas, dollar sign)."
    )

    # Salary year is in allowed range
    salary_year_leaf = evaluator.add_leaf(
        id=f"{(d.district_name or 'unknown').lower().replace(' ', '_')}_salary_year_allowed",
        desc=f"{label}: Salary year is within 2023–24, 2024, or 2025 per the cited salary source(s)",
        parent=node,
        critical=True,
    )
    claim_salary_year = (
        f"The cited superintendent salary for {label} corresponds to year '{d.salary_year_label}', "
        f"which is within {ALLOWED_SALARY_YEARS_TEXT}."
    )
    await evaluator.verify(
        claim=claim_salary_year,
        node=salary_year_leaf,
        sources=d.salary_urls,
        additional_instruction="If the label uses formats like 'FY 2025' or '2024-25', consider whether it falls within 2023–24, 2024, or 2025."
    )


async def verify_reporting_for_qualifying_district(
    evaluator: Evaluator,
    parent_node,
    d: DistrictData
) -> None:
    """
    Under Node 2 (reporting requirements), for each listed qualifying district,
    verify that all required data items are provided and supported by sources.
    """
    label = district_label(d.district_name or "Unknown District", d.state)
    base_id = (d.district_name or "unknown").lower().replace(" ", "_")

    node = evaluator.add_sequential(
        id=f"reporting_{base_id}",
        desc=f"Reporting completeness and source support for {label}",
        parent=parent_node,
        critical=True
    )

    # Existence and completeness check: all required data and sources are present
    required_present = (
        has_digits(d.enrollment) and
        urls_nonempty(d.enrollment_urls) and
        has_digits(d.salary) and
        (d.salary_year_label is not None and str(d.salary_year_label).strip() != "") and
        urls_nonempty(d.salary_urls)
    )
    evaluator.add_custom_node(
        result=required_present,
        id=f"{base_id}_required_fields_present",
        desc=f"{label}: Required fields present (numeric enrollment, enrollment source(s), numeric salary, salary year label, salary source(s))",
        parent=node,
        critical=True
    )

    # Enrollment exact figure supported by sources
    enroll_exact_leaf = evaluator.add_leaf(
        id=f"{base_id}_enrollment_supported",
        desc=f"{label}: Enrollment figure is supported by cited enrollment source(s)",
        parent=node,
        critical=True
    )
    claim_enroll_exact = (
        f"The current student enrollment of {label} is '{d.enrollment}' "
        f"(labeled as {d.enrollment_year_label or 'most recent'})."
    )
    await evaluator.verify(
        claim=claim_enroll_exact,
        node=enroll_exact_leaf,
        sources=d.enrollment_urls,
        additional_instruction="Verify that the cited source(s) support the stated enrollment (accept rounding or approximate phrasing if clearly equivalent)."
    )

    # Enrollment threshold > 40,000 supported by same sources
    enroll_over_leaf = evaluator.add_leaf(
        id=f"{base_id}_enrollment_gt_40k_supported",
        desc=f"{label}: Enrollment > 40,000 supported by cited enrollment source(s)",
        parent=node,
        critical=True
    )
    claim_enroll_over = (
        f"The current student enrollment for {label} exceeds 40,000 students "
        f"(based on the same cited enrollment source[s])."
    )
    await evaluator.verify(
        claim=claim_enroll_over,
        node=enroll_over_leaf,
        sources=d.enrollment_urls,
        additional_instruction="Confirm that the cited enrollment implies > 40,000, allowing standard rounding tolerances."
    )

    # Salary exact amount and year supported by sources
    salary_exact_leaf = evaluator.add_leaf(
        id=f"{base_id}_salary_supported",
        desc=f"{label}: Superintendent salary amount and year supported by cited salary source(s)",
        parent=node,
        critical=True
    )
    claim_salary_exact = (
        f"The superintendent of {label} has an annual salary of '{d.salary}' for the year '{d.salary_year_label}'."
    )
    await evaluator.verify(
        claim=claim_salary_exact,
        node=salary_exact_leaf,
        sources=d.salary_urls,
        additional_instruction="Verify that both the amount and the explicit year label are supported by the cited source(s)."
    )

    # Salary year within allowed range (also supported by source)
    salary_year_leaf = evaluator.add_leaf(
        id=f"{base_id}_salary_year_allowed",
        desc=f"{label}: Salary year is within {ALLOWED_SALARY_YEARS_TEXT} per salary source(s)",
        parent=node,
        critical=True
    )
    claim_salary_year = (
        f"The stated salary year for {label} ('{d.salary_year_label}') is within {ALLOWED_SALARY_YEARS_TEXT}."
    )
    await evaluator.verify(
        claim=claim_salary_year,
        node=salary_year_leaf,
        sources=d.salary_urls,
        additional_instruction="Consider typical labeling variants such as '2024-25', 'FY 2025', or '2023–24' to determine if the year falls within the allowed set."
    )

    # Salary threshold ≥ $300,000 supported by salary sources
    salary_ge_leaf = evaluator.add_leaf(
        id=f"{base_id}_salary_ge_300k_supported",
        desc=f"{label}: Salary ≥ $300,000 supported by cited salary source(s)",
        parent=node,
        critical=True
    )
    claim_salary_ge = (
        f"The superintendent's annual salary for {label} is at least $300,000."
    )
    await evaluator.verify(
        claim=claim_salary_ge,
        node=salary_ge_leaf,
        sources=d.salary_urls,
        additional_instruction="Verify from the cited source(s) that the salary meets or exceeds $300,000."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the district selection + reporting task.
    """
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

    # Add canonical info to summary for context
    evaluator.add_ground_truth({
        "allowed_districts": canonical_names(),
        "salary_years_allowed": ALLOWED_SALARY_YEARS_TEXT
    }, gt_type="task_constraints")

    # Extraction
    extraction: DistrictsExtraction = await evaluator.extract(
        prompt=prompt_extract_districts(),
        template_class=DistrictsExtraction,
        extraction_name="districts_extraction"
    )

    # Build a consistent list of qualifying districts, filtered to the five canonical names (preserve answer order)
    allowed_set = {n.lower(): n for n in canonical_names()}
    extracted_qualifying: List[str] = []
    for name in (extraction.qualifying_districts or []):
        if isinstance(name, str) and name.strip().lower() in allowed_set:
            extracted_qualifying.append(allowed_set[name.strip().lower()])
        else:
            # Keep as-is to let the membership check catch outside entries
            if isinstance(name, str) and name.strip():
                extracted_qualifying.append(name.strip())

    # Construct a mapping for quick record lookup
    rec_map: Dict[str, DistrictData] = {}
    for cn in canonical_names():
        rec = find_record_for_canonical(extraction, cn)
        if not rec:
            # Create a placeholder record if missing
            state = next((d["state"] for d in CANONICAL_DISTRICTS if d["name"] == cn), None)
            rec = DistrictData(district_name=cn, state=state)
        rec_map[cn] = rec

    # Create a task-level critical node (since initialize() root is non-critical by design)
    task_root = evaluator.add_parallel(
        id="task_root",
        desc=("Determine, among the five specified districts, which ones meet BOTH conditions (enrollment > 40,000; "
              "superintendent salary ≥ $300,000 with salary year in 2023–24, 2024, or 2025), and verify the reported "
              "figures and sources for each district listed as qualifying."),
        parent=root,
        critical=True
    )

    # Node 1: Selection exactness (approximation via: subset check + truth of constraints for all listed)
    selection_node = evaluator.add_parallel(
        id="selection_is_exact_subset",
        desc=("The set of districts identified as meeting the criteria is limited to the five named districts, and "
              "each listed district does satisfy BOTH constraints per cited sources."),
        parent=task_root,
        critical=True
    )

    # Leaf: membership subset of five
    await verify_selection_subset_of_five(evaluator, selection_node, extracted_qualifying)

    # For each listed qualifying district, verify BOTH constraints are supported by sources
    for qname in extracted_qualifying:
        # If the name is not among the canonical five, still attempt to create a placeholder record
        if qname in rec_map:
            rec = rec_map[qname]
        else:
            rec = DistrictData(district_name=qname, state=None)
        await verify_listed_district_meets_both(evaluator, selection_node, rec)

    # Node 2: Reporting completeness and source grounding for each listed qualifying district
    reporting_node = evaluator.add_parallel(
        id="reporting_for_each_listed_qualifier",
        desc=("For every district the answer lists as qualifying, it provides a numeric current enrollment figure "
              "(2025–26 or most recent), a numeric superintendent salary figure, the salary year (2023–24, 2024, or 2025), "
              "and verifiable sources (URLs) for both enrollment and salary; and these are supported by the sources."),
        parent=task_root,
        critical=True
    )

    for qname in extracted_qualifying:
        if qname in rec_map:
            rec = rec_map[qname]
        else:
            rec = DistrictData(district_name=qname, state=None)
        await verify_reporting_for_qualifying_district(evaluator, reporting_node, rec)

    return evaluator.get_summary()