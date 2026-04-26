import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fl_student_planning_2026"
TASK_DESCRIPTION = (
    "A Florida high school junior who is a student-athlete is planning their senior year and college applications for fall 2026 admission. "
    "They want to: (1) Maintain NCAA Division I athletic eligibility, (2) Qualify for the Florida Bright Futures Florida Academic Scholars (FAS) award, "
    "(3) Take dual enrollment courses at a Florida community college during their senior year, and (4) Apply to Yale University using Single-Choice Early Action. "
    "For this student, identify and provide the following specific requirements: A. NCAA Division I Eligibility Requirements; B. Florida Bright Futures FAS Requirements; "
    "C. Dual Enrollment Eligibility (Florida Community College); D. Yale Single-Choice Early Action Application. Provide each answer with a reference URL that supports the information."
)

# Ground-truth reference (expected policy values)
GROUND_TRUTH = {
    "NCAA": {
        "total_core_courses": "16",
        "pre_senior_core_courses": "10",
        "pre_senior_core_subjects_required": "7 in English, Mathematics, or Science",
        "core_gpa_minimum": "2.3",
        "test_scores_required": "Not required (eliminated in January 2023)"
    },
    "BrightFuturesFAS": {
        "weighted_gpa_minimum": "3.5 (weighted)",
        "sat_required_score": "SAT 1330 (or ACT 29)",
        "service_hours_required": "100 volunteer hours OR 100 paid work hours OR a combination"
    },
    "DualEnrollment": {
        "min_unweighted_gpa": "Typically 3.0 (some programs accept 2.5)",
        "min_course_grade_to_continue": "C or higher"
    },
    "YaleSCEA": {
        "application_deadline_date": "November 1",
        "decision_notification_date": "Mid-December (e.g., December 17, 2025 at 5pm ET for Class of 2030)",
        "restrictions_policy": "Cannot apply ED or REA/SCEA to other universities; EA to other schools may be permitted per Yale policy"
    }
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class NCAAInfo(BaseModel):
    total_core_courses: Optional[str] = None
    total_core_courses_urls: List[str] = Field(default_factory=list)

    pre_senior_core_courses: Optional[str] = None  # e.g., "10"
    pre_senior_core_subjects_required: Optional[str] = None  # e.g., "7 in English, Mathematics, or Science"
    pre_senior_urls: List[str] = Field(default_factory=list)

    core_gpa_minimum: Optional[str] = None  # e.g., "2.3"
    gpa_urls: List[str] = Field(default_factory=list)

    test_scores_required: Optional[str] = None  # e.g., "Not required"
    test_urls: List[str] = Field(default_factory=list)


class BrightFuturesFASInfo(BaseModel):
    weighted_gpa_minimum: Optional[str] = None  # e.g., "3.5"
    bf_gpa_urls: List[str] = Field(default_factory=list)

    sat_required_score: Optional[str] = None  # e.g., "1330 (ACT 29)"
    bf_test_urls: List[str] = Field(default_factory=list)

    service_hours_required: Optional[str] = None  # e.g., "100 hours or combination"
    bf_hours_urls: List[str] = Field(default_factory=list)


class DualEnrollmentInfo(BaseModel):
    min_unweighted_gpa: Optional[str] = None  # e.g., "3.0 (some accept 2.5)"
    de_gpa_urls: List[str] = Field(default_factory=list)

    min_course_grade_to_continue: Optional[str] = None  # e.g., "C or higher"
    de_grade_urls: List[str] = Field(default_factory=list)


class YaleSCEAInfo(BaseModel):
    application_deadline_date: Optional[str] = None  # e.g., "November 1"
    deadline_urls: List[str] = Field(default_factory=list)

    decision_notification_date: Optional[str] = None  # e.g., "Mid-December; Dec 17, 2025 5pm ET (Class of 2030)"
    notification_urls: List[str] = Field(default_factory=list)

    restrictions_policy: Optional[str] = None  # e.g., "No ED/REA to other; EA allowed per policy"
    restrictions_urls: List[str] = Field(default_factory=list)


class RequirementsExtraction(BaseModel):
    ncaa: Optional[NCAAInfo] = None
    bright_futures_fas: Optional[BrightFuturesFASInfo] = None
    dual_enrollment: Optional[DualEnrollmentInfo] = None
    yale_scea: Optional[YaleSCEAInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    Extract the specific requirements and their supporting URLs as presented in the answer. Organize the output into four categories:
    1) ncaa
    2) bright_futures_fas
    3) dual_enrollment
    4) yale_scea

    For each category, extract the following fields exactly as stated in the answer (use strings for values), and collect all explicitly cited URLs that support each item. If URLs are mentioned in markdown, extract the actual URL. If a field is missing, set it to null; if URLs are missing, return an empty list for that URL field.

    A. NCAA Division I Eligibility:
      - total_core_courses (string, e.g., "16")
      - total_core_courses_urls (array of URLs supporting the total core course requirement)
      - pre_senior_core_courses (string, e.g., "10")
      - pre_senior_core_subjects_required (string, e.g., "7 in English, Mathematics, or Science")
      - pre_senior_urls (array of URLs supporting pre-senior requirements)
      - core_gpa_minimum (string, e.g., "2.3")
      - gpa_urls (array of URLs supporting the GPA requirement)
      - test_scores_required (string, e.g., "Not required" or "Required")
      - test_urls (array of URLs supporting the test score requirement status)

    B. Florida Bright Futures Florida Academic Scholars (FAS):
      - weighted_gpa_minimum (string, e.g., "3.5")
      - bf_gpa_urls (array of URLs supporting the GPA requirement)
      - sat_required_score (string, e.g., "1330 (ACT 29)")
      - bf_test_urls (array of URLs supporting the test score requirement)
      - service_hours_required (string, e.g., "100 volunteer hours or 100 paid work hours or combination")
      - bf_hours_urls (array of URLs supporting service/work hours requirement)

    C. Dual Enrollment (Florida Community College):
      - min_unweighted_gpa (string, e.g., "3.0 (some programs accept 2.5)")
      - de_gpa_urls (array of URLs supporting the GPA requirement)
      - min_course_grade_to_continue (string, e.g., "C or higher")
      - de_grade_urls (array of URLs supporting the grade requirement)

    D. Yale Single-Choice Early Action (SCEA):
      - application_deadline_date (string, e.g., "November 1")
      - deadline_urls (array of URLs supporting the deadline)
      - decision_notification_date (string, e.g., "Mid-December; December 17, 2025 at 5pm ET for Class of 2030")
      - notification_urls (array of URLs supporting decision notification timing)
      - restrictions_policy (string, e.g., "Cannot apply ED or REA/SCEA to other universities; can apply Early Action to other schools")
      - restrictions_urls (array of URLs supporting the restriction policy)
    """


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_ncaa_verification(evaluator: Evaluator, parent_node, info: Optional[NCAAInfo]) -> None:
    ncaa_node = evaluator.add_parallel(
        id="NCAA_Division_I_Requirements",
        desc="All NCAA Division I eligibility requirements are correctly identified",
        parent=parent_node,
        critical=False
    )

    # Core Course Requirements
    core_courses_node = evaluator.add_parallel(
        id="Core_Course_Requirements",
        desc="NCAA core course completion requirements",
        parent=ncaa_node,
        critical=True
    )

    # Reference URL for Core Courses (existence check)
    core_ref_exists = evaluator.add_custom_node(
        result=bool(info and info.total_core_courses_urls),
        id="Reference_URL_Core_Courses",
        desc="Valid reference URL provided for core course requirements",
        parent=core_courses_node,
        critical=True
    )

    # Total Core Courses = 16
    total_core_leaf = evaluator.add_leaf(
        id="Total_Core_Courses",
        desc="Total number of core courses required is 16",
        parent=core_courses_node,
        critical=True
    )
    await evaluator.verify(
        claim="For NCAA Division I eligibility, students must complete 16 core courses.",
        node=total_core_leaf,
        sources=(info.total_core_courses_urls if info else []),
        extra_prerequisites=[core_ref_exists],
        additional_instruction="Verify this requirement on NCAA official or authoritative eligibility pages. Allow 'core-course' wording variants."
    )

    # Pre-senior year requirements
    pre_senior_node = evaluator.add_parallel(
        id="Pre_Senior_Year_Courses",
        desc="Number of core courses that must be completed before senior year",
        parent=core_courses_node,
        critical=True
    )

    # Reference URL existence for pre-senior
    pre_ref_exists = evaluator.add_custom_node(
        result=bool(info and info.pre_senior_urls),
        id="Reference_URL_Pre_Senior",
        desc="Valid reference URL provided for pre-senior year course requirements",
        parent=pre_senior_node,
        critical=True
    )

    # Ten courses before senior
    ten_before_leaf = evaluator.add_leaf(
        id="Ten_Courses_Before_Senior",
        desc="10 core courses must be completed before the start of 12th grade",
        parent=pre_senior_node,
        critical=True
    )
    await evaluator.verify(
        claim="For NCAA Division I, 10 core courses must be completed before the start of 12th grade.",
        node=ten_before_leaf,
        sources=(info.pre_senior_urls if info else []),
        extra_prerequisites=[pre_ref_exists],
        additional_instruction="Confirm the '10 core courses before senior year' policy from NCAA resources."
    )

    # Seven in English/Math/Science
    seven_subject_leaf = evaluator.add_leaf(
        id="Seven_Core_Subject_Courses",
        desc="Of the 10 pre-senior year courses, 7 must be in English, Mathematics, or Science",
        parent=pre_senior_node,
        critical=True
    )
    await evaluator.verify(
        claim="Of the 10 pre-senior year core courses, 7 must be in English, Mathematics, or Science.",
        node=seven_subject_leaf,
        sources=(info.pre_senior_urls if info else []),
        extra_prerequisites=[pre_ref_exists],
        additional_instruction="Verify this distribution requirement (7 courses in English/Math/Science) on NCAA sources."
    )

    # Academic Standards
    academic_node = evaluator.add_parallel(
        id="Academic_Standards",
        desc="NCAA academic performance standards",
        parent=ncaa_node,
        critical=True
    )

    # Core GPA requirement
    gpa_req_node = evaluator.add_parallel(
        id="Core_GPA_Requirement",
        desc="Minimum core-course GPA requirement",
        parent=academic_node,
        critical=True
    )
    gpa_ref_exists = evaluator.add_custom_node(
        result=bool(info and info.gpa_urls),
        id="Reference_URL_GPA",
        desc="Valid reference URL provided for GPA requirement",
        parent=gpa_req_node,
        critical=True
    )
    gpa_leaf = evaluator.add_leaf(
        id="GPA_Value_2_3",
        desc="Minimum core-course GPA is 2.3 (or 2.300)",
        parent=gpa_req_node,
        critical=True
    )
    await evaluator.verify(
        claim="The minimum NCAA Division I core-course GPA requirement is 2.3 (2.300).",
        node=gpa_leaf,
        sources=(info.gpa_urls if info else []),
        extra_prerequisites=[gpa_ref_exists],
        additional_instruction="Verify the minimum core-course GPA threshold on NCAA eligibility resources."
    )

    # Test score requirement
    test_req_node = evaluator.add_parallel(
        id="Test_Score_Requirement",
        desc="Standardized test score requirements for NCAA",
        parent=academic_node,
        critical=True
    )
    test_ref_exists = evaluator.add_custom_node(
        result=bool(info and info.test_urls),
        id="Reference_URL_Test",
        desc="Valid reference URL provided for test requirement status",
        parent=test_req_node,
        critical=True
    )
    test_leaf = evaluator.add_leaf(
        id="No_Test_Required",
        desc="SAT/ACT test scores are NOT required (eliminated in January 2023)",
        parent=test_req_node,
        critical=True
    )
    await evaluator.verify(
        claim="SAT or ACT test scores are not required for NCAA Division I initial eligibility (requirement eliminated in January 2023).",
        node=test_leaf,
        sources=(info.test_urls if info else []),
        extra_prerequisites=[test_ref_exists],
        additional_instruction="Confirm the test score requirement status change (no longer required) on NCAA official announcements or eligibility pages."
    )


async def build_bright_futures_verification(evaluator: Evaluator, parent_node, info: Optional[BrightFuturesFASInfo]) -> None:
    bf_node = evaluator.add_parallel(
        id="Florida_Bright_Futures_FAS_Requirements",
        desc="All Florida Bright Futures Florida Academic Scholars requirements are correctly identified",
        parent=parent_node,
        critical=False
    )

    # GPA requirement
    bf_gpa_node = evaluator.add_parallel(
        id="GPA_Requirement",
        desc="Minimum weighted GPA requirement for FAS",
        parent=bf_node,
        critical=True
    )
    bf_gpa_ref = evaluator.add_custom_node(
        result=bool(info and info.bf_gpa_urls),
        id="Reference_URL_BF_GPA",
        desc="Valid reference URL provided for Bright Futures GPA requirement",
        parent=bf_gpa_node,
        critical=True
    )
    bf_gpa_leaf = evaluator.add_leaf(
        id="GPA_Value_3_5",
        desc="Minimum weighted GPA is 3.5",
        parent=bf_gpa_node,
        critical=True
    )
    await evaluator.verify(
        claim="For the Florida Bright Futures Florida Academic Scholars (FAS) award, the minimum weighted GPA requirement is 3.5.",
        node=bf_gpa_leaf,
        sources=(info.bf_gpa_urls if info else []),
        extra_prerequisites=[bf_gpa_ref],
        additional_instruction="Verify FAS GPA requirements on official Florida Bright Futures or Florida Department of Education pages."
    )

    # Test score requirement
    bf_test_node = evaluator.add_parallel(
        id="Test_Score_Requirement",
        desc="Standardized test score requirement for FAS",
        parent=bf_node,
        critical=True
    )
    bf_test_ref = evaluator.add_custom_node(
        result=bool(info and info.bf_test_urls),
        id="Reference_URL_BF_Test",
        desc="Valid reference URL provided for Bright Futures test score requirement",
        parent=bf_test_node,
        critical=True
    )
    bf_test_leaf = evaluator.add_leaf(
        id="SAT_Score_1330",
        desc="Required SAT score is 1330 (or ACT 29)",
        parent=bf_test_node,
        critical=True
    )
    await evaluator.verify(
        claim="For the Florida Bright Futures FAS award, the required standardized test score is SAT 1330 or ACT composite 29.",
        node=bf_test_leaf,
        sources=(info.bf_test_urls if info else []),
        extra_prerequisites=[bf_test_ref],
        additional_instruction="Confirm the SAT/ACT thresholds for FAS using Florida Bright Futures official resources."
    )

    # Service hours requirement
    bf_hours_node = evaluator.add_parallel(
        id="Service_Hours_Requirement",
        desc="Volunteer service or paid work hours requirement",
        parent=bf_node,
        critical=True
    )
    bf_hours_ref = evaluator.add_custom_node(
        result=bool(info and info.bf_hours_urls),
        id="Reference_URL_BF_Hours",
        desc="Valid reference URL provided for service/work hours requirement",
        parent=bf_hours_node,
        critical=True
    )
    bf_hours_leaf = evaluator.add_leaf(
        id="Hours_100",
        desc="100 volunteer service hours, 100 paid work hours, or combination required",
        parent=bf_hours_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Florida Bright Futures FAS award requires 100 volunteer service hours, 100 paid work hours, or a combination to meet the service/work requirement.",
        node=bf_hours_leaf,
        sources=(info.bf_hours_urls if info else []),
        extra_prerequisites=[bf_hours_ref],
        additional_instruction="Verify service/work hour requirements on official Florida Bright Futures guidance."
    )


async def build_dual_enrollment_verification(evaluator: Evaluator, parent_node, info: Optional[DualEnrollmentInfo]) -> None:
    de_node = evaluator.add_parallel(
        id="Dual_Enrollment_Requirements",
        desc="Dual enrollment eligibility requirements at Florida community colleges",
        parent=parent_node,
        critical=False
    )

    # GPA requirement
    de_gpa_node = evaluator.add_parallel(
        id="GPA_Requirement",
        desc="Minimum unweighted high school GPA for dual enrollment eligibility",
        parent=de_node,
        critical=True
    )
    de_gpa_ref = evaluator.add_custom_node(
        result=bool(info and info.de_gpa_urls),
        id="Reference_URL_DE_GPA",
        desc="Valid reference URL provided for dual enrollment GPA requirement",
        parent=de_gpa_node,
        critical=True
    )
    de_gpa_leaf = evaluator.add_leaf(
        id="GPA_3_0_Typical",
        desc="Minimum unweighted GPA is typically 3.0 (some programs accept 2.5)",
        parent=de_gpa_node,
        critical=True
    )
    await evaluator.verify(
        claim="For Florida community college dual enrollment, the minimum unweighted high school GPA is typically 3.0 (some programs accept 2.5).",
        node=de_gpa_leaf,
        sources=(info.de_gpa_urls if info else []),
        extra_prerequisites=[de_gpa_ref],
        additional_instruction="Use Florida DOE or specific college dual enrollment pages; some programs list 2.5 as acceptable."
    )

    # Course grade requirement
    de_grade_node = evaluator.add_parallel(
        id="Course_Grade_Requirement",
        desc="Minimum grade to maintain in dual enrollment courses",
        parent=de_node,
        critical=True
    )
    de_grade_ref = evaluator.add_custom_node(
        result=bool(info and info.de_grade_urls),
        id="Reference_URL_DE_Grade",
        desc="Valid reference URL provided for dual enrollment grade requirement",
        parent=de_grade_node,
        critical=True
    )
    de_grade_leaf = evaluator.add_leaf(
        id="Grade_C_Minimum",
        desc="Must earn a grade of C or higher in dual enrollment courses to continue",
        parent=de_grade_node,
        critical=True
    )
    await evaluator.verify(
        claim="Students must earn a grade of C or higher in dual enrollment courses to continue or remain eligible.",
        node=de_grade_leaf,
        sources=(info.de_grade_urls if info else []),
        extra_prerequisites=[de_grade_ref],
        additional_instruction="Verify minimum grade continuation policy using Florida community college dual enrollment guidelines."
    )


async def build_yale_scea_verification(evaluator: Evaluator, parent_node, info: Optional[YaleSCEAInfo]) -> None:
    yale_node = evaluator.add_parallel(
        id="Yale_SCEA_Requirements",
        desc="Yale Single-Choice Early Action application requirements and timeline",
        parent=parent_node,
        critical=False
    )

    # Application Timeline
    timeline_node = evaluator.add_parallel(
        id="Application_Timeline",
        desc="Yale SCEA application deadlines and notification dates",
        parent=yale_node,
        critical=True
    )

    # Deadline
    deadline_node = evaluator.add_parallel(
        id="Application_Deadline",
        desc="SCEA application deadline information",
        parent=timeline_node,
        critical=True
    )
    deadline_ref = evaluator.add_custom_node(
        result=bool(info and info.deadline_urls),
        id="Reference_URL_Yale_Deadline",
        desc="Valid reference URL provided for Yale SCEA deadline",
        parent=deadline_node,
        critical=True
    )
    deadline_leaf = evaluator.add_leaf(
        id="Deadline_November_1",
        desc="Application deadline is November 1",
        parent=deadline_node,
        critical=True
    )
    await evaluator.verify(
        claim="Yale's Single-Choice Early Action application deadline for fall 2026 admission is November 1.",
        node=deadline_leaf,
        sources=(info.deadline_urls if info else []),
        extra_prerequisites=[deadline_ref],
        additional_instruction="Verify deadline on Yale's official undergraduate admissions website."
    )

    # Notification
    notify_node = evaluator.add_parallel(
        id="Decision_Notification",
        desc="Expected decision notification date",
        parent=timeline_node,
        critical=True
    )
    notify_ref = evaluator.add_custom_node(
        result=bool(info and info.notification_urls),
        id="Reference_URL_Yale_Notification",
        desc="Valid reference URL provided for Yale notification date",
        parent=notify_node,
        critical=True
    )
    notify_leaf = evaluator.add_leaf(
        id="Notification_Mid_December",
        desc="Decision notification is in mid-December (specifically December 17, 2025 at 5pm ET for Class of 2030)",
        parent=notify_node,
        critical=True
    )
    await evaluator.verify(
        claim="Yale releases SCEA decisions in mid-December; for the Class of 2030 cycle, decisions were released on December 17, 2025 at 5pm Eastern.",
        node=notify_leaf,
        sources=(info.notification_urls if info else []),
        extra_prerequisites=[notify_ref],
        additional_instruction="Confirm decision release timing using Yale Admissions communications or official pages."
    )

    # Application Restrictions
    restrict_node = evaluator.add_parallel(
        id="Application_Restrictions",
        desc="Restrictions on applying to other universities under Yale SCEA",
        parent=yale_node,
        critical=True
    )
    restrict_ref = evaluator.add_custom_node(
        result=bool(info and info.restrictions_urls),
        id="Reference_URL_Yale_Restrictions",
        desc="Valid reference URL provided for Yale SCEA restrictions",
        parent=restrict_node,
        critical=True
    )
    restrict_leaf = evaluator.add_leaf(
        id="SCEA_Restrictions",
        desc="Cannot apply Early Decision or Restrictive/Single-Choice Early Action to other universities; can apply regular Early Action to other schools",
        parent=restrict_node,
        critical=True
    )
    await evaluator.verify(
        claim="Under Yale's Single-Choice Early Action policy, applicants cannot apply Early Decision or Restrictive/Single-Choice Early Action to other universities; applying to other non-binding Early Action programs may be permitted as described by Yale.",
        node=restrict_leaf,
        sources=(info.restrictions_urls if info else []),
        extra_prerequisites=[restrict_ref],
        additional_instruction="Verify Yale's SCEA restrictions on the official Yale Admissions policy page."
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
    Evaluate the answer for the Florida student-athlete planning task across NCAA, Bright Futures, Dual Enrollment, and Yale SCEA.
    """
    # Initialize evaluator
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

    # Extract structured requirements from the answer
    extracted: RequirementsExtraction = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=RequirementsExtraction,
        extraction_name="requirements_extraction"
    )

    # Add ground truth info
    evaluator.add_ground_truth({"expected_requirements": GROUND_TRUTH}, gt_type="ground_truth_requirements")

    # Top-level aggregation node (non-critical to allow partial credit across categories)
    top = evaluator.add_parallel(
        id="Complete_Requirements_Analysis",
        desc="Verification of all requirements for a Florida student-athlete's college planning across NCAA, Bright Futures, Dual Enrollment, and Yale SCEA",
        parent=root,
        critical=False
    )

    # Build sub-verifications
    await build_ncaa_verification(evaluator, top, extracted.ncaa)
    await build_bright_futures_verification(evaluator, top, extracted.bright_futures_fas)
    await build_dual_enrollment_verification(evaluator, top, extracted.dual_enrollment)
    await build_yale_scea_verification(evaluator, top, extracted.yale_scea)

    # Return structured summary
    return evaluator.get_summary()