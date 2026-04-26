import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nc_district_academics_position"
TASK_DESCRIPTION = """
Identify a district-level administrative position in curriculum, instruction, or academic affairs within a North Carolina public school district that meets the following requirements:

1. The school district must have a student enrollment of at least 50,000 students
2. The position must be a district-level (central office) administrative role, not a school-based position such as principal or assistant principal
3. The position must be specifically focused on curriculum, instruction, or academic affairs (not operations, facilities, finance, or transportation)
4. The position must require a minimum of a master's degree in education, educational leadership, curriculum and instruction, or a closely related field
5. The position must require at least 3 years of relevant professional experience in education
6. The position must require or strongly prefer North Carolina administrator licensure from the NC Department of Public Instruction
7. The position must offer an annual salary of at least $80,000
8. The position must be associated with the 2025-2026 or 2026-2027 school year (either currently posted, recently filled, or anticipated)

Provide the following information:
- The school district name and its enrollment size
- The complete job title of the position
- The minimum educational requirement
- The minimum experience requirement
- The licensure requirement
- The salary or salary range
- The timeframe/school year
- A brief description of the key responsibilities
- URL references for: district enrollment data, job posting or position description, and salary information
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PositionExtraction(BaseModel):
    district_name: Optional[str] = None
    enrollment: Optional[str] = None

    position_title: Optional[str] = None
    education_requirement: Optional[str] = None
    experience_requirement: Optional[str] = None
    licensure_requirement: Optional[str] = None
    licensure_level: Optional[str] = None  # e.g., principal, superintendent (if specified)

    salary: Optional[str] = None
    timeframe: Optional[str] = None  # e.g., "2025-2026", "2026-2027"
    responsibilities: Optional[str] = None  # brief summary text

    enrollment_urls: List[str] = Field(default_factory=list)
    position_urls: List[str] = Field(default_factory=list)
    salary_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_position_info() -> str:
    return """
    Extract the following fields from the answer if they are explicitly present. Do not infer or invent information.

    Required textual fields (use exact text from the answer; if missing, return null):
    - district_name: The official name of the North Carolina public school district
    - enrollment: The stated enrollment number or range for the district (e.g., "160,000", "approximately 140k", "over 50,000")
    - position_title: The complete official job title of the position
    - education_requirement: The minimum educational degree requirement (copy exact phrasing)
    - experience_requirement: The minimum experience requirement (copy exact phrasing)
    - licensure_requirement: The licensure requirement or preference (copy exact phrasing)
    - licensure_level: If the type/level of NC administrator license is specified (e.g., principal, superintendent), extract it; otherwise null
    - salary: The annual salary or salary range or salary grade (copy exact phrasing)
    - timeframe: The school year or timeframe associated with the position, e.g., "2025-2026" or "2026-2027" or wording like "for the 2025-26 school year"
    - responsibilities: A brief description/sentence of key responsibilities as presented in the answer (e.g., “leads curriculum and instruction across the district”)

    URL fields (extract actual URLs exactly as presented; if none, return empty array):
    - enrollment_urls: URLs that support the enrollment data (district website, NC DPI data, reputable third-party profiles)
    - position_urls: URLs to the job posting, position description, org chart, or official announcement
    - salary_urls: URLs that specifically document salary, such as the job posting section showing pay or a district salary schedule/NC DPI salary data

    IMPORTANT:
    - Only extract URLs explicitly present in the answer (plain or in markdown).
    - Do not deduce school year from dates unless it is explicitly stated in the answer.
    - Keep all fields as strings; do not convert to numeric types.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    out = []
    seen = set()
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def _combine_sources(*lists: List[str]) -> List[str]:
    combined: List[str] = []
    seen = set()
    for l in lists:
        for u in l:
            if u and u not in seen:
                combined.append(u)
                seen.add(u)
    return combined


def _parse_first_int(text: Optional[str]) -> Optional[int]:
    """Parse the first reasonable integer-like value in text (handles 160,000, 160k, 1.6m)."""
    if not text:
        return None
    pattern = re.compile(r'(?i)\$?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*([kKmM]?)')
    for m in pattern.finditer(text):
        num_str = m.group(1)
        suffix = m.group(2).lower() if m.group(2) else ""
        try:
            num = float(num_str.replace(",", ""))
            if suffix == 'k':
                num *= 1_000
            elif suffix == 'm':
                num *= 1_000_000
            return int(round(num))
        except Exception:
            continue
    return None


def _parse_min_salary(text: Optional[str]) -> Optional[int]:
    """
    Extract the minimum annual salary from a salary string.
    Handles ranges like "$80,000 - $95,000", "80k–100k".
    Returns None if no annual figure can be found.
    """
    if not text:
        return None
    # Quick reject for hourly-only language
    if re.search(r'(?i)\bper\s+hour|\bhourly\b', text):
        # Not annual; cannot verify threshold reliably
        return None

    # Find all numeric tokens with optional k/m suffix
    nums: List[int] = []
    pattern = re.compile(r'(?i)\$?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*([kKmM]?)')
    for m in pattern.finditer(text):
        num_str = m.group(1)
        suffix = (m.group(2) or "").lower()
        try:
            val = float(num_str.replace(",", ""))
            if suffix == 'k':
                val *= 1_000
            elif suffix == 'm':
                val *= 1_000_000
            nums.append(int(round(val)))
        except Exception:
            continue

    if not nums:
        return None
    return min(nums)


# --------------------------------------------------------------------------- #
# Verification tree construction functions                                    #
# --------------------------------------------------------------------------- #
async def build_district_identification(
    evaluator: Evaluator,
    parent,
    data: PositionExtraction
) -> None:
    node = evaluator.add_parallel(
        id="District_Identification",
        desc="Correctly identifies a North Carolina school district and verifies it meets the enrollment size requirement",
        parent=parent,
        critical=True  # Section should be essential
    )

    # District_Name_And_Location (leaf)
    dn_leaf = evaluator.add_leaf(
        id="District_Name_And_Location",
        desc="Provides the complete official name of the North Carolina school district",
        parent=node,
        critical=True
    )
    district_name = data.district_name or ""
    district_sources = _combine_sources(
        _normalize_urls(data.enrollment_urls),
        _normalize_urls(data.position_urls)
    )
    claim_dn = f"The identified entity '{district_name}' is a public school district in North Carolina."
    await evaluator.verify(
        claim=claim_dn,
        node=dn_leaf,
        sources=district_sources,
        additional_instruction="Verify that the named entity is indeed a North Carolina public school district (LEA). Allow minor naming variants (e.g., CMS for Charlotte-Mecklenburg Schools)."
    )

    # Enrollment_Size_Verification (parallel)
    esv = evaluator.add_parallel(
        id="Enrollment_Size_Verification",
        desc="Verifies the district has student enrollment of at least 50,000 students",
        parent=node,
        critical=True
    )

    # Enrollment_Criteria (parallel)
    ec = evaluator.add_parallel(
        id="Enrollment_Criteria",
        desc="Validates enrollment data and threshold compliance",
        parent=esv,
        critical=True
    )

    # Enrollment_Data_Provided (existence)
    enrollment_provided = evaluator.add_custom_node(
        result=bool(data.enrollment and data.enrollment.strip()),
        id="Enrollment_Data_Provided",
        desc="Provides the actual enrollment number or range for the identified district",
        parent=ec,
        critical=True
    )

    # Minimum_Threshold_Met (computed)
    parsed_enrollment = _parse_first_int(data.enrollment)
    threshold_ok = parsed_enrollment is not None and parsed_enrollment >= 50_000
    evaluator.add_custom_node(
        result=threshold_ok,
        id="Minimum_Threshold_Met",
        desc="The provided enrollment number meets or exceeds 50,000 students",
        parent=ec,
        critical=True
    )

    # Enrollment_Documentation
    ed = evaluator.add_parallel(
        id="Enrollment_Documentation",
        desc="Provides source documentation for enrollment data",
        parent=esv,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_normalize_urls(data.enrollment_urls)) > 0,
        id="Enrollment_Source_Reference",
        desc="Provides URL reference to the source of enrollment data (e.g., district website, NC DPI data, Niche.com ranking)",
        parent=ed,
        critical=True
    )


async def build_position_identification(
    evaluator: Evaluator,
    parent,
    data: PositionExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Position_Identification",
        desc="Correctly identifies a specific administrative position and verifies it meets the position level and focus area requirements",
        parent=parent,
        critical=True
    )

    # Position_Title (verify against posting URL)
    pt_leaf = evaluator.add_leaf(
        id="Position_Title",
        desc="Provides the complete official job title of the administrative position",
        parent=node,
        critical=True
    )
    title = data.position_title or ""
    claim_title = f"The official job title of the position is '{title}'."
    await evaluator.verify(
        claim=claim_title,
        node=pt_leaf,
        sources=_normalize_urls(data.position_urls),
        additional_instruction="Verify the job title as presented on the official posting or district page. Allow minor capitalization or punctuation differences."
    )

    # District_Level_Verification
    dlv = evaluator.add_parallel(
        id="District_Level_Verification",
        desc="Verifies the position is a district-level (central office) role, not a school-based position",
        parent=node,
        critical=True
    )

    # Central_Office_Designation
    cod_leaf = evaluator.add_leaf(
        id="Central_Office_Designation",
        desc="Confirms the position is designated as a district-level or central office administrative role",
        parent=dlv,
        critical=True
    )
    claim_cod = "This position is a district-level (central office) administrative role."
    await evaluator.verify(
        claim=claim_cod,
        node=cod_leaf,
        sources=_normalize_urls(data.position_urls),
        additional_instruction="Look for language such as 'district office', 'central services', 'office of academics', reporting to CAO/COO/Superintendent, or roles that serve the whole district rather than a single school."
    )

    # Not_School_Based
    nsb_leaf = evaluator.add_leaf(
        id="Not_School_Based",
        desc="Confirms the position is not a school-based role such as principal or assistant principal",
        parent=dlv,
        critical=True
    )
    claim_nsb = "This position is not a school-based role (e.g., not a principal or assistant principal) and is not based at a single school."
    await evaluator.verify(
        claim=claim_nsb,
        node=nsb_leaf,
        sources=_normalize_urls(data.position_urls),
        additional_instruction="Ensure the posting does not describe a principal/AP role or single-school position."
    )

    # Focus_Area_Verification
    fav = evaluator.add_parallel(
        id="Focus_Area_Verification",
        desc="Verifies the position is specifically related to curriculum, instruction, or academic affairs",
        parent=node,
        critical=True
    )

    ciaf_leaf = evaluator.add_leaf(
        id="Curriculum_Instruction_Academic_Focus",
        desc="Confirms the position title or description explicitly indicates responsibility for curriculum, instruction, or academic affairs",
        parent=fav,
        critical=True
    )
    claim_ciaf = "The position focuses on curriculum, instruction, teaching and learning, or academic affairs at the district level."
    await evaluator.verify(
        claim=claim_ciaf,
        node=ciaf_leaf,
        sources=_normalize_urls(data.position_urls),
        additional_instruction="Check duties such as leading curriculum frameworks, instructional improvement, academic programs, or content-area leadership."
    )

    nos_leaf = evaluator.add_leaf(
        id="Not_Operations_Support",
        desc="Confirms the position is not primarily focused on operations, facilities, finance, transportation, or other non-academic support areas",
        parent=fav,
        critical=True
    )
    claim_nos = "The position is not primarily an operations/facilities/finance/transportation role; its primary emphasis is academic."
    await evaluator.verify(
        claim=claim_nos,
        node=nos_leaf,
        sources=_normalize_urls(data.position_urls),
        additional_instruction="If the title/description indicates operations or support services as primary function, mark as not satisfied."
    )

    # Position_Documentation
    pd = evaluator.add_parallel(
        id="Position_Documentation",
        desc="Provides source documentation for the position",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_normalize_urls(data.position_urls)) > 0,
        id="Position_Source_Reference",
        desc="Provides URL reference to the job posting, district organizational chart, or official announcement",
        parent=pd,
        critical=True
    )


async def build_education_verification(
    evaluator: Evaluator,
    parent,
    data: PositionExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Educational_Requirement_Verification",
        desc="Verifies the position's educational credential requirements meet the minimum master's degree standard",
        parent=parent,
        critical=True
    )

    # Degree_Requirements
    dr = evaluator.add_parallel(
        id="Degree_Requirements",
        desc="Validates the degree level and field requirements",
        parent=node,
        critical=True
    )

    # Minimum_Degree_Level (existence)
    evaluator.add_custom_node(
        result=bool(data.education_requirement and data.education_requirement.strip()),
        id="Minimum_Degree_Level",
        desc="States the minimum degree requirement for the position",
        parent=dr,
        critical=True
    )

    # Masters_Or_Higher_Required
    moh_leaf = evaluator.add_leaf(
        id="Masters_Or_Higher_Required",
        desc="Confirms the position requires at minimum a master's degree (or higher such as Ed.S. or doctorate)",
        parent=dr,
        critical=True
    )
    claim_moh = "The position requires at least a master's degree or higher (e.g., Ed.S., Ed.D., Ph.D.)."
    await evaluator.verify(
        claim=claim_moh,
        node=moh_leaf,
        sources=_normalize_urls(data.position_urls),
        additional_instruction="Accept phrasing like 'Master’s degree required', 'Master’s preferred but Bachelor’s not acceptable'. If 'preferred' is stated but explicitly says Bachelor's acceptable, do not pass."
    )

    # Field_Specification
    fs_leaf = evaluator.add_leaf(
        id="Field_Specification",
        desc="Confirms the required field is education, educational leadership, curriculum and instruction, or closely related educational field",
        parent=dr,
        critical=True
    )
    claim_fs = "The required field is in education, educational leadership, curriculum & instruction, or a closely related educational field."
    await evaluator.verify(
        claim=claim_fs,
        node=fs_leaf,
        sources=_normalize_urls(data.position_urls),
        additional_instruction="Look for degree fields such as Education, Educational Leadership/Administration, Curriculum & Instruction, Teaching & Learning. Closely related educational fields are acceptable."
    )

    # Education_Documentation
    ed = evaluator.add_parallel(
        id="Education_Documentation",
        desc="Provides source documentation for educational requirements",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_normalize_urls(data.position_urls)) > 0,
        id="Education_Requirement_Source",
        desc="Provides URL reference to job posting or job description documenting the educational requirements",
        parent=ed,
        critical=True
    )


async def build_experience_verification(
    evaluator: Evaluator,
    parent,
    data: PositionExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Experience_Requirement_Verification",
        desc="Verifies the position's experience requirements meet the minimum 3 years standard",
        parent=parent,
        critical=True
    )

    ec = evaluator.add_parallel(
        id="Experience_Criteria",
        desc="Validates the experience duration and type requirements",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(data.experience_requirement and data.experience_requirement.strip()),
        id="Minimum_Years_Required",
        desc="States the minimum number of years of experience required for the position",
        parent=ec,
        critical=True
    )

    three_leaf = evaluator.add_leaf(
        id="Three_Years_Or_More",
        desc="Confirms the position requires at least 3 years of relevant professional experience",
        parent=ec,
        critical=True
    )
    claim_three = "The position requires at least three (3) years of relevant professional experience."
    await evaluator.verify(
        claim=claim_three,
        node=three_leaf,
        sources=_normalize_urls(data.position_urls),
        additional_instruction="Accept wording such as '3 years minimum', 'three years of administrative experience required', etc."
    )

    et_leaf = evaluator.add_leaf(
        id="Experience_Type",
        desc="Confirms the required experience type is in education (teaching, administration, or educational leadership)",
        parent=ec,
        critical=True
    )
    claim_et = "The required experience is in education (e.g., teaching, school/district administration, or educational leadership)."
    await evaluator.verify(
        claim=claim_et,
        node=et_leaf,
        sources=_normalize_urls(data.position_urls),
        additional_instruction="Check for phrases like 'K-12 teaching experience', 'school administration', 'district leadership', or similar."
    )

    ed = evaluator.add_parallel(
        id="Experience_Documentation",
        desc="Provides source documentation for experience requirements",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_normalize_urls(data.position_urls)) > 0,
        id="Experience_Requirement_Source",
        desc="Provides URL reference to job posting or job description documenting the experience requirements",
        parent=ed,
        critical=True
    )


async def build_licensure_verification(
    evaluator: Evaluator,
    parent,
    data: PositionExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Licensure_Requirement_Verification",
        desc="Verifies the position requires or prefers North Carolina administrator licensure",
        parent=parent,
        critical=True
    )

    lc = evaluator.add_parallel(
        id="Licensure_Criteria",
        desc="Validates the licensure requirements",
        parent=node,
        # NOTE: To comply with framework constraints (critical parent cannot have non-critical child),
        # we mark this aggregator critical=True and its children critical=True.
        # This slightly strengthens the requirement for 'Licensure_Level_Type'.
        critical=True
    )

    nc_leaf = evaluator.add_leaf(
        id="NC_Administrator_License_Specified",
        desc="Confirms the position explicitly requires or prefers North Carolina administrator licensure issued by NC DPI",
        parent=lc,
        critical=True
    )
    claim_nc = "The position requires or strongly prefers a North Carolina administrator license issued by the NC Department of Public Instruction."
    await evaluator.verify(
        claim=claim_nc,
        node=nc_leaf,
        sources=_normalize_urls(data.position_urls),
        additional_instruction="Accept phrasing like 'NC administrator license', 'North Carolina school administrator licensure', 'NC DPI administrator license', etc."
    )

    # Set as critical to satisfy parent constraint; the claim will be flexible and pass if evidence shows a specific level.
    level_leaf = evaluator.add_leaf(
        id="Licensure_Level_Type",
        desc="Specifies the type or level of NC administrator license required (e.g., principal license, superintendent license, or general administrator license)",
        parent=lc,
        critical=True
    )
    claim_level = "The posting specifies the type or level of the North Carolina administrator license (e.g., principal, superintendent, curriculum/academics administrator) or indicates a general administrator license."
    await evaluator.verify(
        claim=claim_level,
        node=level_leaf,
        sources=_normalize_urls(data.position_urls),
        additional_instruction="Pass if the posting names a specific NC administrator license type or clearly indicates a general administrator license. If no mention at all, fail."
    )

    ld = evaluator.add_parallel(
        id="Licensure_Documentation",
        desc="Provides source documentation for licensure requirements",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_normalize_urls(data.position_urls)) > 0,
        id="Licensure_Requirement_Source",
        desc="Provides URL reference to job posting or job description documenting the licensure requirements",
        parent=ld,
        critical=True
    )


async def build_salary_verification(
    evaluator: Evaluator,
    parent,
    data: PositionExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Salary_Verification",
        desc="Verifies the position's compensation meets the minimum $80,000 annual salary requirement",
        parent=parent,
        critical=True
    )

    sc = evaluator.add_parallel(
        id="Salary_Criteria",
        desc="Validates salary information and threshold compliance",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(data.salary and data.salary.strip()),
        id="Salary_Information_Provided",
        desc="Provides the annual salary, salary range, or salary grade for the position",
        parent=sc,
        critical=True
    )

    min_salary = _parse_min_salary(data.salary)
    evaluator.add_custom_node(
        result=(min_salary is not None and min_salary >= 80_000),
        id="Minimum_Salary_Threshold_Met",
        desc="Confirms the stated salary or minimum of the salary range is at least $80,000 annually",
        parent=sc,
        critical=True
    )

    sd = evaluator.add_parallel(
        id="Salary_Documentation",
        desc="Provides source documentation for salary information",
        parent=node,
        critical=True
    )
    all_salary_sources = _normalize_urls(data.salary_urls) or _normalize_urls(data.position_urls)
    evaluator.add_custom_node(
        result=len(all_salary_sources) > 0,
        id="Salary_Source_Reference",
        desc="Provides URL reference to job posting, district salary schedule, or NC DPI salary data documenting the compensation",
        parent=sd,
        critical=True
    )


async def build_temporal_verification(
    evaluator: Evaluator,
    parent,
    data: PositionExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Temporal_Validity",
        desc="Verifies the position is relevant to the 2025-2026 or 2026-2027 school year timeframe",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(data.timeframe and data.timeframe.strip()),
        id="Timeframe_Specified",
        desc="Clearly states when the position is available, was posted, or was filled",
        parent=node,
        critical=True
    )

    cr_leaf = evaluator.add_leaf(
        id="Current_Relevance",
        desc="Confirms the position is associated with the 2025-2026 or 2026-2027 school year (currently posted, recently filled, or anticipated opening)",
        parent=node,
        critical=True
    )
    tf = data.timeframe or ""
    claim_cr = ("The position is associated with the 2025-2026 or 2026-2027 school year "
                "(e.g., explicitly mentions 2025-26 or 2026-27, is currently posted for that year, "
                "or is announced as an anticipated opening for those years).")
    await evaluator.verify(
        claim=claim_cr,
        node=cr_leaf,
        sources=_normalize_urls(data.position_urls),
        additional_instruction="Accept variants like '2025-26', 'SY 2026-27'. If dates indicate posting or duties tied to those school years, pass."
    )


async def build_responsibilities(
    evaluator: Evaluator,
    parent,
    data: PositionExtraction
) -> None:
    node = evaluator.add_leaf(
        id="Job_Responsibilities_Description",
        desc="Provides a brief description of the key job responsibilities that confirms the position's focus on curriculum/instruction/academic affairs",
        parent=parent,
        critical=False  # Non-critical per rubric
    )
    responsibilities = (data.responsibilities or "").strip()
    claim_resp = (f"Key responsibilities for the position include: {responsibilities}. "
                  f"These responsibilities confirm the position focuses on curriculum, instruction, or academic affairs.")
    await evaluator.verify(
        claim=claim_resp,
        node=node,
        sources=_normalize_urls(data.position_urls),
        additional_instruction="Pass if the posting’s duties align with curriculum, instruction, teaching & learning, or academic program oversight."
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
    Evaluate an answer for the NC district-level academics position task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates sections independently
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

    # Extract structured info
    extracted: PositionExtraction = await evaluator.extract(
        prompt=prompt_extract_position_info(),
        template_class=PositionExtraction,
        extraction_name="position_extraction"
    )

    # Normalize URL lists
    extracted.enrollment_urls = _normalize_urls(extracted.enrollment_urls)
    extracted.position_urls = _normalize_urls(extracted.position_urls)
    extracted.salary_urls = _normalize_urls(extracted.salary_urls)

    # Add some custom info for debugging/visibility
    evaluator.add_custom_info(
        info={
            "parsed_enrollment_int": _parse_first_int(extracted.enrollment),
            "parsed_min_salary_int": _parse_min_salary(extracted.salary),
            "district_name": extracted.district_name,
            "position_title": extracted.position_title
        },
        info_type="parsed_fields",
        info_name="parsed_fields_summary"
    )

    # Build the top-level rubric node (non-critical to allow optional sub-criteria)
    top = evaluator.add_parallel(
        id="Position_Identification_And_Validation",
        desc="Identifies and validates a district-level administrative position in curriculum/instruction/academic affairs in a large North Carolina school district that meets all specified requirements",
        parent=root,
        critical=False  # Keep non-critical to allow non-critical child (responsibilities) without violating constraints
    )

    # Build sub-sections
    await build_district_identification(evaluator, top, extracted)
    await build_position_identification(evaluator, top, extracted)
    await build_education_verification(evaluator, top, extracted)
    await build_experience_verification(evaluator, top, extracted)
    await build_licensure_verification(evaluator, top, extracted)
    await build_salary_verification(evaluator, top, extracted)
    await build_temporal_verification(evaluator, top, extracted)
    await build_responsibilities(evaluator, top, extracted)

    return evaluator.get_summary()