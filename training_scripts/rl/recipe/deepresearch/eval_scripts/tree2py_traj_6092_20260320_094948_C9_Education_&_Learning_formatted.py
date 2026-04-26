import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cwru_ccp_level1_stem_courses"
TASK_DESCRIPTION = (
    "An Ohio high school student in 11th grade is planning to participate in Case Western Reserve "
    "University's College Credit Plus (CCP) program for the 2026-2027 academic year and wants to take STEM courses "
    "that will count toward their first 15 college credits while ensuring these credits transfer to any Ohio public "
    "university. Identify 4 distinct STEM courses offered by Case Western Reserve University that meet ALL of the "
    "following requirements: (1) Ohio CCP Level 1 Eligibility - The course must be designated as a Level 1 course under "
    "Ohio's College Credit Plus program, meaning it qualifies for a student's first 15 college credits and must be "
    "transferable under Ohio Transfer Module (OTM), Transfer Assurance Guides (TAG), Career-Technical Assurance Guides "
    "(CTAG), or equivalent transfer agreements; (2) CWRU Availability - The course must be offered by Case Western "
    "Reserve University and available to College Credit Plus students; (3) Transfer Guarantee - The course must have "
    "guaranteed transfer status among Ohio public institutions through TAG, OTM, CTAG designation, or equivalent, and "
    "must fall under an approved transfer category such as Mathematics, Natural Sciences, or other OTM/TAG approved "
    "categories; (4) High School Prerequisites - The course prerequisites must be achievable through standard high "
    "school coursework available to students in grades 7-12, and the course cannot require completion of prior "
    "college-level courses beyond what a new CCP student entering the program could reasonably have completed; "
    "(5) STEM Classification - The course must be classified under a STEM department at CWRU, including but not limited "
    "to Mathematics, Computer Science, Engineering Science, Physics, Chemistry, Biology, or related STEM disciplines. "
    "For each of the 4 courses, provide the complete course code and title, a description of how the course meets each "
    "requirement, and reference URLs from CWRU course catalogs/schedules and Ohio transfer credit resources confirming "
    "the course details, prerequisites, and transfer status."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CourseItem(BaseModel):
    course_code: Optional[str] = None
    course_title: Optional[str] = None
    cwru_course_urls: List[str] = Field(default_factory=list)
    transfer_urls: List[str] = Field(default_factory=list)
    credit_hours: Optional[str] = None  # e.g., "3 Units", "4 credits", etc.
    prerequisites: Optional[str] = None
    department: Optional[str] = None  # e.g., "Mathematics", "Physics", etc.
    department_url: Optional[str] = None
    transfer_category: Optional[str] = None  # e.g., "OTM Mathematics (TMM001)", "Natural Sciences (TNS)", etc.


class CoursesExtraction(BaseModel):
    courses: List[CourseItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_courses() -> str:
    return """
Extract up to all STEM courses the answer proposes for Case Western Reserve University's College Credit Plus (CCP) program.

For each proposed course, extract the following fields:
- course_code: The full official course code (e.g., MATH 121, PHYS 121).
- course_title: The official course title.
- cwru_course_urls: A list of URL(s) from official CWRU sources (catalog/bulletin/schedule/department pages) that describe the course and/or its prerequisites and credit units. Include only real URLs explicitly present in the answer.
- transfer_urls: A list of URL(s) from official Ohio statewide transfer resources (e.g., ohiohighered.org, transfercredit.ohio.gov) that document OTM/TAG/CTAG or equivalent state transfer guarantees relevant to this course (or its standard Ohio equivalency). Include only real URLs explicitly present in the answer.
- credit_hours: The credit hours/units wording as shown (e.g., "3 Units", "4 credits"). If phrased differently (e.g., Units), extract it literally.
- prerequisites: The prerequisite text as shown on the CWRU page or as quoted in the answer (e.g., 'three years of high school mathematics' or 'MATH placement' or 'CHEM placement' or 'none').
- department: The CWRU department or program offering the course (e.g., Mathematics, Physics, Computer & Data Sciences, Engineering).
- department_url: A department page URL (if present in the answer) that supports the department classification. If not provided, set to null.
- transfer_category: If provided in the answer, the named OTM/TAG/CTAG area/category or code (e.g., 'OTM Mathematics', 'TMM001', 'Natural Sciences'). If not provided, set to null.

GENERAL RULES:
- Do not invent URLs or fields. Extract only what is explicitly present in the answer text.
- Prefer CWRU official domains for cwru_course_urls such as 'case.edu', 'casewesternreserve.edu', 'cwru.edu', 'bulletin.case.edu', 'catalog.case.edu'.
- Prefer Ohio statewide transfer domains for transfer_urls such as 'ohiohighered.org', 'transfercredit.ohio.gov', or subdomains thereof.
- If any field is missing for a course, set it to null or an empty list as appropriate.

Return JSON with a single key 'courses' containing an array of these course objects.
    """


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def _first_n_distinct_courses(courses: List[CourseItem], k: int = 4) -> List[CourseItem]:
    seen = set()
    result: List[CourseItem] = []
    for c in courses:
        key = (c.course_code or "").strip().lower() + "||" + (c.course_title or "").strip().lower()
        if key not in seen and (c.course_code or c.course_title):
            seen.add(key)
            result.append(c)
        if len(result) >= k:
            break
    return result


def _pad_to_k(courses: List[CourseItem], k: int = 4) -> List[CourseItem]:
    while len(courses) < k:
        courses.append(CourseItem())
    return courses


def _safe_list(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        v = u.strip()
        if v and v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _is_cwru_url(url: str) -> bool:
    u = (url or "").lower()
    return any(dom in u for dom in ["case.edu", "casewesternreserve.edu", "cwru.edu", "bulletin.case.edu", "catalog.case.edu"])


def _has_any_cwru_url(urls: List[str]) -> bool:
    return any(_is_cwru_url(u) for u in urls)


def _is_ohio_transfer_url(url: str) -> bool:
    u = (url or "").lower()
    return any(dom in u for dom in ["ohiohighered.org", "transfercredit.ohio.gov", "ohio.gov"])


def _has_any_ohio_transfer_url(urls: List[str]) -> bool:
    return any(_is_ohio_transfer_url(u) for u in urls)


def _full_course_label(course: CourseItem) -> str:
    code = (course.course_code or "").strip()
    title = (course.course_title or "").strip()
    if code and title:
        return f"{code} {title}"
    return code or title or "the course"


# --------------------------------------------------------------------------- #
# Verification logic per course                                               #
# --------------------------------------------------------------------------- #
async def verify_one_course(evaluator: Evaluator, parent_node, course: CourseItem, idx: int) -> None:
    course_label = _full_course_label(course)
    cwru_urls = _safe_list(course.cwru_course_urls)
    transfer_urls = _safe_list(course.transfer_urls)
    dept_url = [course.department_url] if course.department_url else []
    all_dept_sources = _safe_list(cwru_urls + dept_url)

    # Create Course node (parallel, non-critical to allow partial credit per course)
    course_node = evaluator.add_parallel(
        id=f"course_{idx+1}",
        desc=f"{['First', 'Second', 'Third', 'Fourth'][idx] if idx < 4 else f'Course {idx+1}'} eligible STEM course meeting all requirements",
        parent=parent_node,
        critical=False,
    )

    # ---------------- CCP Eligibility ----------------
    ccp_node = evaluator.add_parallel(
        id=f"course_{idx+1}_ccp_eligibility",
        desc="Verify the course meets Ohio College Credit Plus Level 1 eligibility requirements",
        parent=course_node,
        critical=True,
    )

    # 1) Ohio_CCP_Compliance
    ohio_ccp_node = evaluator.add_parallel(
        id=f"course_{idx+1}_ohio_ccp_compliance",
        desc="Course satisfies Ohio CCP program requirements",
        parent=ccp_node,
        critical=True,
    )

    # Gate: require at least one credible Ohio transfer URL for the CCP/transfer-related checks
    transfer_ref_exists = evaluator.add_custom_node(
        result=_has_any_ohio_transfer_url(transfer_urls),
        id=f"course_{idx+1}_transfer_ref_exists",
        desc="Has at least one official Ohio transfer URL (ohiohighered.org / transfercredit.ohio.gov)",
        parent=ohio_ccp_node,
        critical=True,
    )

    # Level_1_Status (Critical)
    level1_leaf = evaluator.add_leaf(
        id=f"course_{idx+1}_level_1_status",
        desc="Course is designated as Level 1 (eligible for first 15 credits) under Ohio CCP, via OTM/TAG/CTAG transferability",
        parent=ohio_ccp_node,
        critical=True,
    )
    level1_claim = (
        "This course qualifies as an Ohio CCP Level I course for the student's first 15 credit hours "
        "because it is officially transferable under the Ohio Transfer Module (OTM), Transfer Assurance Guides (TAG), "
        "Career-Technical Assurance Guides (CTAG), or an equivalent statewide transfer designation."
    )
    await evaluator.verify(
        claim=level1_claim,
        node=level1_leaf,
        sources=transfer_urls,
        additional_instruction=(
            "Decide ONLY based on the provided transfer URL(s). If the page(s) show OTM/TAG/CTAG or equivalent statewide "
            "transfer designation for the course (or clearly equivalent content), consider this Level I compliant. "
            "If no valid statewide transfer page is provided, mark as not supported."
        ),
        extra_prerequisites=[transfer_ref_exists],
    )

    # Credit_Hours (Critical)
    credit_leaf = evaluator.add_leaf(
        id=f"course_{idx+1}_credit_hours",
        desc="Course carries college credit hours/units (typically 3–4 for STEM courses)",
        parent=ohio_ccp_node,
        critical=True,
    )
    credit_claim = (
        f"The official CWRU page for {course_label} shows that the course carries college credit (units/credits) "
        "and lists the credit amount (e.g., 'Units: 3', '3 credits')."
    )
    await evaluator.verify(
        claim=credit_claim,
        node=credit_leaf,
        sources=cwru_urls,
        additional_instruction=(
            "Pass if the CWRU catalog/bulletin/schedule page explicitly lists credit hours/units. "
            "CWRU often uses the term 'Units'."
        ),
    )

    # 2) CWRU_Program_Availability
    cwru_avail_node = evaluator.add_parallel(
        id=f"course_{idx+1}_cwru_program_availability",
        desc="Course is available through CWRU's CCP program",
        parent=ccp_node,
        critical=True,
    )

    # Gate: require at least one credible CWRU course URL
    cwru_ref_exists = evaluator.add_custom_node(
        result=_has_any_cwru_url(cwru_urls),
        id=f"course_{idx+1}_cwru_ref_exists",
        desc="Has at least one official CWRU course/catalog/schedule URL",
        parent=cwru_avail_node,
        critical=True,
    )

    # Course_Offering (Critical)
    offering_leaf = evaluator.add_leaf(
        id=f"course_{idx+1}_course_offering",
        desc="Course is offered by CWRU and available to CCP students",
        parent=cwru_avail_node,
        critical=True,
    )
    offering_claim = (
        f"The provided CWRU page(s) show that {course_label} is an official undergraduate course offered by "
        "Case Western Reserve University. Because CCP Level I courses must be transferable and the course is offered "
        "by CWRU, it is available to CCP students subject to admissions/placement."
    )
    await evaluator.verify(
        claim=offering_claim,
        node=offering_leaf,
        sources=cwru_urls,
        additional_instruction=(
            "Treat 'offered by CWRU' as satisfied if the page is an official CWRU catalog/bulletin/schedule page for "
            "the course. Do not require the page to explicitly say 'CCP'; Level I + CWRU offering is sufficient "
            "for availability."
        ),
        extra_prerequisites=[cwru_ref_exists],
    )

    # Course_Reference (Critical) — verify that at least one provided URL is a valid official CWRU course reference
    course_ref_leaf = evaluator.add_leaf(
        id=f"course_{idx+1}_course_reference",
        desc="Provide a valid reference URL from CWRU course catalog or schedule confirming the course details",
        parent=cwru_avail_node,
        critical=True,
    )
    course_ref_claim = (
        "At least one of the provided CWRU URLs is an official course catalog/bulletin/schedule page that confirms the "
        f"course details for {course_label}."
    )
    await evaluator.verify(
        claim=course_ref_claim,
        node=course_ref_leaf,
        sources=cwru_urls,
        additional_instruction=(
            "Confirm that at least one URL is a legitimate CWRU official page (e.g., bulletin.case.edu, catalog.case.edu, "
            "case.edu, cwru.edu) describing this course."
        ),
        extra_prerequisites=[cwru_ref_exists],
    )

    # ---------------- Transfer Status ----------------
    transfer_status_node = evaluator.add_parallel(
        id=f"course_{idx+1}_transfer_status",
        desc="Verify the course has guaranteed transfer status among Ohio public institutions",
        parent=course_node,
        critical=True,
    )

    # State_Transfer_Guarantee
    state_transfer_node = evaluator.add_parallel(
        id=f"course_{idx+1}_state_transfer_guarantee",
        desc="Course has official Ohio transfer designation",
        parent=transfer_status_node,
        critical=True,
    )

    # TAG_OTM_Designation (Critical)
    tag_otm_leaf = evaluator.add_leaf(
        id=f"course_{idx+1}_tag_otm_designation",
        desc="Course is part of Ohio Transfer Module, Transfer Assurance Guide, or Career-Technical Assurance Guide",
        parent=state_transfer_node,
        critical=True,
    )
    tag_claim = (
        "The provided Ohio transfer URL(s) show that the course (or a clear Ohio statewide equivalent for its content) "
        "has an official OTM, TAG, or CTAG designation (e.g., OTM Mathematics, TAG Calculus TMM001, or similar)."
    )
    await evaluator.verify(
        claim=tag_claim,
        node=tag_otm_leaf,
        sources=transfer_urls,
        additional_instruction=(
            "Accept if the page explicitly shows an OTM/TAG/CTAG designation for the same content (e.g., TMM00x for calculus). "
            "Reject if the URL is not an official Ohio statewide transfer page or does not indicate a designation."
        ),
        extra_prerequisites=[transfer_ref_exists],
    )

    # Subject_Classification (Critical)
    subj_class_leaf = evaluator.add_leaf(
        id=f"course_{idx+1}_subject_classification",
        desc="Course falls under Mathematics, Natural Sciences, or other OTM/TAG approved category",
        parent=state_transfer_node,
        critical=True,
    )
    subj_cat = (course.transfer_category or "").strip()
    subj_class_claim = (
        "The official Ohio transfer documentation classifies this course in an approved OTM/TAG area such as Mathematics, "
        f"Natural Sciences, or another statewide-approved category. Stated category (if provided): '{subj_cat}'."
    )
    await evaluator.verify(
        claim=subj_class_claim,
        node=subj_class_leaf,
        sources=transfer_urls,
        additional_instruction=(
            "Focus on the category/area shown on the Ohio transfer page(s). Accept standard approved areas like Mathematics, "
            "Natural Science, etc."
        ),
        extra_prerequisites=[transfer_ref_exists],
    )

    # Transfer_Documentation
    transfer_doc_node = evaluator.add_parallel(
        id=f"course_{idx+1}_transfer_documentation",
        desc="Transfer status is officially documented",
        parent=transfer_status_node,
        critical=True,
    )

    # Transfer_Reference (Critical) — existence/validity of official statewide transfer URL(s)
    transfer_ref_leaf = evaluator.add_custom_node(
        result=_has_any_ohio_transfer_url(transfer_urls),
        id=f"course_{idx+1}_transfer_reference",
        desc="Provide a valid reference URL confirming the transfer designation status (official Ohio statewide transfer resource)",
        parent=transfer_doc_node,
        critical=True,
    )

    # ---------------- Prerequisites ----------------
    prereq_node = evaluator.add_parallel(
        id=f"course_{idx+1}_prerequisites",
        desc="Verify the course prerequisites are achievable for high school students in grades 7-12",
        parent=course_node,
        critical=True,
    )

    hs_ready_node = evaluator.add_parallel(
        id=f"course_{idx+1}_high_school_readiness",
        desc="Prerequisites can be satisfied through high school preparation",
        parent=prereq_node,
        critical=True,
    )

    # High_School_Preparation (Critical)
    hs_prep_leaf = evaluator.add_leaf(
        id=f"course_{idx+1}_high_school_preparation",
        desc="Prerequisites can be met through high school coursework (e.g., algebra/precalculus, HS science) or placement",
        parent=hs_ready_node,
        critical=True,
    )
    hs_prep_claim = (
        f"The official CWRU page for {course_label} indicates prerequisites that can be met via standard high school "
        f"coursework or placement (e.g., Algebra II/Precalculus, HS chemistry/physics, or an appropriate placement exam). "
        f"Prerequisites text (if provided): '{(course.prerequisites or '').strip()}'."
    )
    await evaluator.verify(
        claim=hs_prep_claim,
        node=hs_prep_leaf,
        sources=cwru_urls,
        additional_instruction=(
            "Accept if prerequisites mention high-school-level preparation, math/science placement, standardized placement, "
            "or similar. Reject if it requires prior college-level courses that a new CCP student could not reasonably have."
        ),
        extra_prerequisites=[cwru_ref_exists],
    )

    # No_College_Prerequisites (Critical)
    no_college_leaf = evaluator.add_leaf(
        id=f"course_{idx+1}_no_college_prereqs",
        desc="Course does not require completion of prior college-level courses beyond what a new CCP student could have",
        parent=hs_ready_node,
        critical=True,
    )
    no_college_claim = (
        f"The CWRU page for {course_label} shows no prerequisite that strictly requires prior college-level coursework "
        "beyond typical high-school preparation or placement."
    )
    await evaluator.verify(
        claim=no_college_claim,
        node=no_college_leaf,
        sources=cwru_urls,
        additional_instruction=(
            "If any prerequisite is a numbered college course (e.g., MATH 122) with no alternative HS/placement route, reject. "
            "If a numbered course is listed but can be waived by placement/AP/HS background, accept."
        ),
        extra_prerequisites=[cwru_ref_exists],
    )

    # Prerequisite_Documentation
    prereq_doc_node = evaluator.add_parallel(
        id=f"course_{idx+1}_prereq_documentation",
        desc="Prerequisites are officially documented",
        parent=prereq_node,
        critical=True,
    )

    # Prerequisite_Reference (Critical)
    prereq_ref_leaf = evaluator.add_leaf(
        id=f"course_{idx+1}_prereq_reference",
        desc="Provide a valid reference URL documenting the course prerequisites",
        parent=prereq_doc_node,
        critical=True,
    )
    prereq_ref_claim = (
        f"The provided CWRU course page(s) for {course_label} explicitly document prerequisites (or state 'none')."
    )
    await evaluator.verify(
        claim=prereq_ref_claim,
        node=prereq_ref_leaf,
        sources=cwru_urls,
        additional_instruction=(
            "Confirm that the CWRU page lists a 'Prerequisite(s)' section or otherwise clearly states the prerequisites. "
            "If the page states that there are no prerequisites, that also counts as documented."
        ),
        extra_prerequisites=[cwru_ref_exists],
    )

    # ---------------- Subject Area (STEM) ----------------
    subject_area_node = evaluator.add_parallel(
        id=f"course_{idx+1}_subject_area",
        desc="Verify the course is in a STEM discipline (Science, Technology, Engineering, Mathematics)",
        parent=course_node,
        critical=True,
    )

    stem_class_node = evaluator.add_parallel(
        id=f"course_{idx+1}_stem_dept_classification",
        desc="Course belongs to a recognized STEM department",
        parent=subject_area_node,
        critical=True,
    )

    # STEM_Classification (Critical)
    stem_class_leaf = evaluator.add_leaf(
        id=f"course_{idx+1}_stem_classification",
        desc="Course is classified under a STEM department at CWRU (Math, CS, Engineering, Physics, Chemistry, Biology, etc.)",
        parent=stem_class_node,
        critical=True,
    )
    dept_name = (course.department or "").strip()
    stem_claim = (
        f"The official CWRU page(s) indicate that {course_label} is offered by a recognized STEM department at CWRU. "
        f"Department (if provided): '{dept_name}'."
    )
    await evaluator.verify(
        claim=stem_claim,
        node=stem_class_leaf,
        sources=all_dept_sources if all_dept_sources else cwru_urls,
        additional_instruction=(
            "Recognize STEM departments such as Mathematics, Physics, Chemistry, Biology, Computer & Data Sciences, "
            "Biomedical Engineering, Electrical Engineering & Computer Science, and similar. "
            "Accept if the page clearly places the course within one of these departments."
        ),
        extra_prerequisites=[cwru_ref_exists],
    )

    # Department_Reference (Critical)
    dept_ref_leaf = evaluator.add_leaf(
        id=f"course_{idx+1}_department_reference",
        desc="Provide a valid reference URL confirming the STEM department classification",
        parent=stem_class_node,
        critical=True,
    )
    dept_ref_claim = (
        f"The provided URL(s) for {course_label} confirm the department classification (either on the course page or a "
        "linked official department page)."
    )
    await evaluator.verify(
        claim=dept_ref_claim,
        node=dept_ref_leaf,
        sources=all_dept_sources if all_dept_sources else cwru_urls,
        additional_instruction=(
            "Confirm that at least one page clearly indicates the hosting department/program for the course and that it "
            "is a STEM department. Accept either the course catalog page (if it names the department) or the official "
            "department site as the confirming reference."
        ),
        extra_prerequisites=[cwru_ref_exists],
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
    Evaluate an answer for the CWRU CCP Level 1 STEM courses transferability task.
    """
    # Initialize evaluator; root is non-critical by default and will aggregate children
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

    # Create a top-level parallel node for the whole task (non-critical to allow partial credit)
    task_node = evaluator.add_parallel(
        id="stem_course_identification_task",
        desc="Identify 4 distinct STEM courses at CWRU that are CCP Level 1 and have statewide transfer guarantees with HS-feasible prerequisites",
        parent=root,
        critical=False,
    )

    # Extract courses from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_courses(),
        template_class=CoursesExtraction,
        extraction_name="extracted_courses",
    )

    # Prepare up to 4 distinct courses
    distinct_courses = _first_n_distinct_courses(extracted.courses or [], 4)
    final_courses = _pad_to_k(distinct_courses, 4)

    # Build verification subtrees per course
    for i, course in enumerate(final_courses[:4]):
        await verify_one_course(evaluator, task_node, course, i)

    # Return standardized evaluation summary
    return evaluator.get_summary()