import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "uc_ncaa_course_plan"
TASK_DESCRIPTION = (
    "You are a college planning counselor helping a California high school student-athlete who plans to apply to "
    "University of California schools and compete in NCAA Division I sports. Create a four-year high school course "
    "plan (grades 9-12) that satisfies both UC A-G admission requirements for California residents and NCAA Division I "
    "initial-eligibility requirements.\n\nYour course plan must satisfy the following requirements:\n\n"
    "UC A-G Requirements (California Residents):\n"
    "- Complete 15 yearlong A-G courses with grade C or better\n"
    "- Complete at least 11 A-G courses before the start of senior year (12th grade)\n"
    "- Achieve minimum 3.0 GPA in A-G courses\n"
    "- Include 2 years of History: one year of world history/cultures/historical geography AND one year of U.S. history or one-half year U.S. history plus one-half year civics/American government\n"
    "- Include 4 years of college-preparatory English\n"
    "- Include 3 years of Mathematics at Algebra I or higher level (must include geometry or integrated math with geometry content)\n"
    "- Include 2 years of Laboratory Science covering two of: biology, chemistry, or physics\n"
    "- Include 2 years of the same Language other than English\n"
    "- Include 1 year of Visual and Performing Arts\n"
    "- Include 1 year of College-preparatory Elective (from Area G or beyond A-F)\n"
    "- All courses must be on the high school's UC-approved A-G course list\n\n"
    "NCAA Division I Core Course Requirements:\n"
    "- Complete 16 core courses\n"
    "- Complete at least 10 core courses before the start of seventh semester (senior year), with at least 7 of those 10 in English, Math, or Natural/Physical Science\n"
    "- Achieve minimum 2.3 GPA in core courses\n"
    "- Include 4 years of English\n"
    "- Include 3 years of Mathematics at Algebra I or higher level\n"
    "- Include 2 years of Natural/Physical Science (with 1 year of lab if offered by the school)\n"
    "- Include 2 years of Social Science\n"
    "- Include 4 additional years of English, Math, or Science\n"
    "- Include 1 additional year from English, Math, Science, Social Science, World Language, Comparative Religion, or Philosophy\n"
    "- All courses must be on the high school's NCAA-approved core course list\n"
    "- Only courses completed in the first 8 semesters from the start of 9th grade count toward these requirements\n\n"
    "Additional Considerations:\n"
    "- Courses that satisfy both UC A-G and NCAA requirements must appear on both the school's UC-approved A-G list and NCAA-approved core course list\n"
    "- You may assume the high school offers standard college-preparatory courses that are approved by both UC and NCAA\n"
    "- Present your course plan organized by grade level (9th, 10th, 11th, 12th)\n"
    "- For each course, provide: course name, which requirement(s) it satisfies (UC A-G area and/or NCAA core area), and assumed letter grade (for GPA calculation)\n"
    "- Provide reference URLs to verify the UC A-G and NCAA Division I requirements you used"
)


# ----------------------------- Data Models --------------------------------- #
class CourseItem(BaseModel):
    course_name: Optional[str] = None
    uc_ag_area: Optional[str] = None  # One of A,B,C,D,E,F,G (or None)
    ncaa_core_area: Optional[str] = None  # One of: English, Math, Natural/Physical Science, Social Science, World Language, Comparative Religion, Philosophy
    letter_grade: Optional[str] = None  # e.g., A, A-, B+, C, etc.
    is_yearlong: Optional[bool] = None
    is_lab: Optional[bool] = None  # for sciences if noted
    science_discipline: Optional[str] = None  # biology, chemistry, physics if applicable
    language_name: Optional[str] = None  # e.g., Spanish, French, Chinese


class GradePlan(BaseModel):
    grade_level: Optional[str] = None  # "9th", "10th", "11th", "12th"
    courses: List[CourseItem] = Field(default_factory=list)


class CoursePlanExtraction(BaseModel):
    grade9: Optional[GradePlan] = None
    grade10: Optional[GradePlan] = None
    grade11: Optional[GradePlan] = None
    grade12: Optional[GradePlan] = None
    uc_reference_urls: List[str] = Field(default_factory=list)
    ncaa_reference_urls: List[str] = Field(default_factory=list)
    states_uc_approved: Optional[bool] = None  # whether the answer explicitly states UC-approved list condition
    states_ncaa_approved: Optional[bool] = None  # whether the answer explicitly states NCAA-approved list condition


# --------------------------- Extraction Prompt ----------------------------- #
def prompt_extract_course_plan() -> str:
    return (
        "Extract the four-year high school course plan organized by grades 9, 10, 11, and 12 from the answer. "
        "Return a JSON object with fields grade9, grade10, grade11, grade12, each containing: "
        "grade_level (e.g., '9th', '10th', '11th', '12th') and an array 'courses'. "
        "For each course, extract:\n"
        "- course_name: the full course name.\n"
        "- uc_ag_area: one of 'A','B','C','D','E','F','G' if the course satisfies a UC A-G area, or null if not indicated.\n"
        "- ncaa_core_area: one of 'English', 'Math', 'Natural/Physical Science', 'Social Science', 'World Language', 'Comparative Religion', 'Philosophy', or null if not indicated.\n"
        "- letter_grade: the assumed letter grade for GPA calculation (e.g., A, A-, B+, B, B-, C+, C, C-, D, F). If not provided, infer a reasonable value from the answer; otherwise set null.\n"
        "- is_yearlong: true if the course is presented as a yearlong course; if not specified, assume true.\n"
        "- is_lab: for science courses, true if the course includes a lab and the answer suggests that; else false. If not specified, use best judgment.\n"
        "- science_discipline: for UC Area D sciences, extract 'biology', 'chemistry', or 'physics' if clearly indicated; else null.\n"
        "- language_name: for UC Area E languages, extract the language (e.g., 'Spanish', 'French', 'Chinese'); else null.\n\n"
        "Also extract:\n"
        "- uc_reference_urls: list of URLs that the answer provides for UC A-G requirements.\n"
        "- ncaa_reference_urls: list of URLs that the answer provides for NCAA Division I core-course requirements.\n"
        "- states_uc_approved: boolean true if the answer explicitly states that all UC A-G courses listed are on the school's UC-approved A-G list; else false.\n"
        "- states_ncaa_approved: boolean true if the answer explicitly states that all NCAA core courses listed are on the school's NCAA-approved core course list; else false.\n"
        "If any field is missing, set it to null or empty list as appropriate. Do not invent URLs."
    )


# --------------------------- Helper Functions ------------------------------ #
def all_courses_with_grade(plan: CoursePlanExtraction) -> List[Tuple[str, CourseItem]]:
    items: List[Tuple[str, CourseItem]] = []
    for grade_key in ["grade9", "grade10", "grade11", "grade12"]:
        grade: Optional[GradePlan] = getattr(plan, grade_key)
        if grade and grade.courses:
            for c in grade.courses:
                items.append((grade.grade_level or grade_key, c))
    return items


def grade_to_points(letter: Optional[str]) -> Optional[float]:
    if not letter:
        return None
    l = letter.strip().upper()
    mapping = {
        "A+": 4.0, "A": 4.0, "A-": 3.7,
        "B+": 3.3, "B": 3.0, "B-": 2.7,
        "C+": 2.3, "C": 2.0, "C-": 1.7,
        "D+": 1.3, "D": 1.0, "D-": 0.7,
        "F": 0.0
    }
    return mapping.get(l, None)


def letter_is_c_or_better(letter: Optional[str]) -> bool:
    if not letter:
        return False
    first = letter.strip().upper()[:1]
    return first in {"A", "B", "C"}


def count_uc_ag_courses(plan: CoursePlanExtraction) -> int:
    return sum(1 for _, c in all_courses_with_grade(plan) if c.uc_ag_area is not None)


def count_uc_ag_c_or_better(plan: CoursePlanExtraction) -> int:
    return sum(1 for _, c in all_courses_with_grade(plan) if c.uc_ag_area is not None and letter_is_c_or_better(c.letter_grade))


def count_uc_ag_before_senior(plan: CoursePlanExtraction) -> int:
    total = 0
    for grade_key in ["grade9", "grade10", "grade11"]:
        grade: Optional[GradePlan] = getattr(plan, grade_key)
        if grade and grade.courses:
            for c in grade.courses:
                if c.uc_ag_area is not None and letter_is_c_or_better(c.letter_grade):
                    total += 1
    return total


def compute_uc_ag_gpa(plan: CoursePlanExtraction) -> Optional[float]:
    points: List[float] = []
    for _, c in all_courses_with_grade(plan):
        if c.uc_ag_area is not None:
            p = grade_to_points(c.letter_grade)
            if p is not None:
                points.append(p)
    if not points:
        return None
    return sum(points) / len(points)


def compute_ncaa_core_gpa(plan: CoursePlanExtraction) -> Optional[float]:
    points: List[float] = []
    for _, c in all_courses_with_grade(plan):
        if c.ncaa_core_area is not None:
            p = grade_to_points(c.letter_grade)
            if p is not None:
                points.append(p)
    if not points:
        return None
    return sum(points) / len(points)


def uc_subject_distribution_checks(plan: CoursePlanExtraction) -> Dict[str, bool]:
    # 2 years history with specific content
    a_courses = [c for _, c in all_courses_with_grade(plan) if (c.uc_ag_area or "").upper() == "A" and c.course_name]
    names_a = [c.course_name.lower() for c in a_courses]
    has_world = any(("world history" in n) or ("world" in n and "history" in n) or ("historical geography" in n) or ("cultures" in n) for n in names_a)
    has_us_history = any(("u.s. history" in n) or ("us history" in n) or ("united states history" in n) for n in names_a)
    has_civics_or_gov = any(("government" in n) or ("civics" in n) or ("american government" in n) for n in names_a)
    hist_ok = (has_world and has_us_history) or (has_world and (has_civics_or_gov and has_us_history)) or (has_world and has_civics_or_gov and not has_us_history)  # allow half+half variant loosely

    # English 4 years (UC B)
    b_count = sum(1 for _, c in all_courses_with_grade(plan) if (c.uc_ag_area or "").upper() == "B")

    # Math 3 years (UC C) Algebra I or higher
    c_math_courses = [c for _, c in all_courses_with_grade(plan) if (c.uc_ag_area or "").upper() == "C" and c.course_name]
    math_count = len(c_math_courses)
    names_c = [c.course_name.lower() for c in c_math_courses]
    has_geometry = any(("geometry" in n) or ("integrated math" in n) for n in names_c)
    has_alg1_or_higher = any(("algebra i" in n) or ("algebra 1" in n) or ("geometry" in n) or ("algebra ii" in n) or ("algebra 2" in n) or ("precalculus" in n) or ("calculus" in n) for n in names_c)

    # Lab science 2 years across two disciplines (UC D)
    d_science_courses = [c for _, c in all_courses_with_grade(plan) if (c.uc_ag_area or "").upper() == "D" and c.course_name]
    disciplines = set()
    for c in d_science_courses:
        disc = (c.science_discipline or "").lower()
        if not disc:
            name = (c.course_name or "").lower()
            if "bio" in name:
                disc = "biology"
            elif "chem" in name:
                disc = "chemistry"
            elif "phys" in name:
                disc = "physics"
        if disc:
            disciplines.add(disc)
    lab_science_ok = len(d_science_courses) >= 2 and len(disciplines.intersection({"biology", "chemistry", "physics"})) >= 2

    # Language other than English 2 years same language (UC E)
    e_lang_courses = [c for _, c in all_courses_with_grade(plan) if (c.uc_ag_area or "").upper() == "E"]
    lang_counts: Dict[str, int] = {}
    for c in e_lang_courses:
        lang = (c.language_name or "").strip().lower()
        if not lang:
            # Try infer language name from course_name
            name = (c.course_name or "").lower()
            m = re.search(r"(spanish|french|chinese|mandarin|german|japanese|latin|korean|italian|arabic|russian)", name)
            if m:
                lang = m.group(1)
        if lang:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
    two_years_same_language = any(cnt >= 2 for cnt in lang_counts.values())

    # Visual and Performing Arts 1 year (UC F)
    f_count = sum(1 for _, c in all_courses_with_grade(plan) if (c.uc_ag_area or "").upper() == "F")

    # College-prep elective 1 year (UC G)
    g_count = sum(1 for _, c in all_courses_with_grade(plan) if (c.uc_ag_area or "").upper() == "G")

    return {
        "UC_History_2_Years_With_Specified_Content": hist_ok,
        "UC_English_4_Years": b_count >= 4,
        "UC_Math_3_Years_AlgIplus": (math_count >= 3) and has_alg1_or_higher,
        "UC_Math_Includes_Geometry_Content": has_geometry,
        "UC_Lab_Science_2_Years_Two_Disciplines": lab_science_ok,
        "UC_Language_2_Years_Same": two_years_same_language,
        "UC_VPA_1_Year": f_count >= 1,
        "UC_Elective_1_Year_Area_G_Or_Beyond": g_count >= 1,
    }


def ncaa_subject_distribution_checks(plan: CoursePlanExtraction) -> Dict[str, bool]:
    core_courses = [c for _, c in all_courses_with_grade(plan) if c.ncaa_core_area]
    english_count = sum(1 for c in core_courses if (c.ncaa_core_area or "").lower() == "english")
    math_courses = [c for c in core_courses if (c.ncaa_core_area or "").lower() == "math"]
    math_count = len(math_courses)
    math_names = [(c.course_name or "").lower() for c in math_courses]
    has_alg1plus = any(("algebra i" in n) or ("algebra 1" in n) or ("geometry" in n) or ("algebra ii" in n) or ("algebra 2" in n) or ("precalculus" in n) or ("calculus" in n) for n in math_names)
    science_courses = [c for c in core_courses if (c.ncaa_core_area or "").lower() in {"natural/physical science", "science"}]
    science_count = len(science_courses)
    lab_present = any(bool(c.is_lab) for c in science_courses)
    social_science_count = sum(1 for c in core_courses if (c.ncaa_core_area or "").lower() == "social science")
    # Additional EMS 4 calculation: total EMS minus baseline minimum of 9
    ems_total = english_count + math_count + science_count
    additional_ems_ok = (ems_total - 9) >= 4
    # Additional 1 from allowed areas: check presence of at least one in allowed areas (including world language, comparative religion, philosophy, or social science beyond 2)
    allowed_areas_courses = [c for c in core_courses if (c.ncaa_core_area or "").lower() in {"world language", "comparative religion", "philosophy"}]
    social_science_beyond_two = social_science_count >= 3
    additional_one_allowed_ok = (len(allowed_areas_courses) >= 1) or social_science_beyond_two

    return {
        "NCAA_English_4": english_count >= 4,
        "NCAA_Math_3_AlgIplus": (math_count >= 3) and has_alg1plus,
        "NCAA_Science_2_With_Lab_If_Offered": (science_count >= 2) and (lab_present or science_count >= 2),
        "NCAA_Social_Science_2": social_science_count >= 2,
        "NCAA_Additional_4_EMS": additional_ems_ok,
        "NCAA_Additional_1_From_Allowed_Areas": additional_one_allowed_ok,
    }


def ncaa_early_10_and_7_ems(plan: CoursePlanExtraction) -> Tuple[bool, bool, int, int]:
    early_core: List[CourseItem] = []
    for grade_key in ["grade9", "grade10", "grade11"]:
        grade: Optional[GradePlan] = getattr(plan, grade_key)
        if grade and grade.courses:
            for c in grade.courses:
                if c.ncaa_core_area:
                    early_core.append(c)
    total_early = len(early_core)
    first_10 = early_core[:10]
    ems_in_first_10 = sum(1 for c in first_10 if (c.ncaa_core_area or "").lower() in {"english", "math", "natural/physical science", "science"})
    return total_early >= 10, ems_in_first_10 >= 7, total_early, ems_in_first_10


def is_plan_organized_by_grade(plan: CoursePlanExtraction) -> bool:
    for grade_key in ["grade9", "grade10", "grade11", "grade12"]:
        grade: Optional[GradePlan] = getattr(plan, grade_key)
        if not grade or not grade.courses:
            return False
    return True


def per_course_fields_present(plan: CoursePlanExtraction) -> bool:
    # Check each course has course_name and letter_grade, and at least one of uc_ag_area or ncaa_core_area
    for _, c in all_courses_with_grade(plan):
        if not c.course_name or not c.course_name.strip():
            return False
        if not c.letter_grade or not c.letter_grade.strip():
            return False
        if not c.uc_ag_area and not c.ncaa_core_area:
            return False
    return True


# --------------------------- Verification Builders ------------------------- #
async def add_plan_presentation_checks(evaluator: Evaluator, parent_node, plan: CoursePlanExtraction) -> None:
    pres_node = evaluator.add_parallel(
        id="Plan_Presentation",
        desc="Plan is presented in the required format and includes required per-course fields.",
        parent=parent_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=is_plan_organized_by_grade(plan),
        id="Organized_By_Grade_Level",
        desc="Course plan is organized by grade level (9th, 10th, 11th, 12th).",
        parent=pres_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=per_course_fields_present(plan),
        id="Per_Course_Fields_Present",
        desc="For each course, provides course name, which requirement(s) it satisfies, and an assumed letter grade.",
        parent=pres_node,
        critical=True
    )


async def add_uc_ag_checks(evaluator: Evaluator, parent_node, plan: CoursePlanExtraction, logger: logging.Logger) -> None:
    uc_node = evaluator.add_parallel(
        id="UC_AG_Requirements",
        desc="All UC A–G requirements in the prompt are satisfied.",
        parent=parent_node,
        critical=True
    )
    # 15 yearlong A-G C or better
    evaluator.add_custom_node(
        result=count_uc_ag_c_or_better(plan) >= 15,
        id="UC_Total_15_AG_Cplus",
        desc="Includes at least 15 yearlong UC A–G courses with grade C or better.",
        parent=uc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=count_uc_ag_before_senior(plan) >= 11,
        id="UC_11_AG_Before_Senior",
        desc="Includes at least 11 UC A–G courses completed before the start of 12th grade.",
        parent=uc_node,
        critical=True
    )
    uc_gpa = compute_uc_ag_gpa(plan)
    evaluator.add_custom_node(
        result=(uc_gpa is not None and uc_gpa >= 3.0),
        id="UC_AG_GPA_Min_3_0",
        desc="UC A–G GPA is at least 3.0.",
        parent=uc_node,
        critical=True
    )

    subj_node = evaluator.add_parallel(
        id="UC_Subject_Distribution",
        desc="UC A–G subject-area distribution requirements are met.",
        parent=uc_node,
        critical=True
    )
    uc_checks = uc_subject_distribution_checks(plan)
    evaluator.add_custom_node(
        result=uc_checks["UC_History_2_Years_With_Specified_Content"],
        id="UC_History_2_Years_With_Specified_Content",
        desc="Includes 2 years of history: world history and U.S. history/civics requirements satisfied.",
        parent=subj_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=uc_checks["UC_English_4_Years"],
        id="UC_English_4_Years",
        desc="Includes 4 years of college-preparatory English.",
        parent=subj_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=uc_checks["UC_Math_3_Years_AlgIplus"],
        id="UC_Math_3_Years_AlgIplus",
        desc="Includes 3 years of mathematics at Algebra I or higher level.",
        parent=subj_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=uc_checks["UC_Math_Includes_Geometry_Content"],
        id="UC_Math_Includes_Geometry_Content",
        desc="UC A–G math sequence includes geometry or integrated math with geometry content.",
        parent=subj_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=uc_checks["UC_Lab_Science_2_Years_Two_Disciplines"],
        id="UC_Lab_Science_2_Years_Two_Disciplines",
        desc="Includes 2 years of laboratory science covering two of: biology, chemistry, or physics.",
        parent=subj_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=uc_checks["UC_Language_2_Years_Same"],
        id="UC_Language_2_Years_Same",
        desc="Includes 2 years of the same language other than English.",
        parent=subj_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=uc_checks["UC_VPA_1_Year"],
        id="UC_VPA_1_Year",
        desc="Includes 1 year of Visual and Performing Arts.",
        parent=subj_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=uc_checks["UC_Elective_1_Year_Area_G_Or_Beyond"],
        id="UC_Elective_1_Year_Area_G_Or_Beyond",
        desc="Includes 1 year of college-preparatory elective from UC A–G Area G (or beyond A–F as stated).",
        parent=subj_node,
        critical=True
    )

    # UC-approved list assertion
    evaluator.add_custom_node(
        result=bool(plan.states_uc_approved),
        id="UC_All_AG_Courses_UC_Approved",
        desc="All UC A–G courses listed are on the high school's UC-approved A–G course list (explicitly stated).",
        parent=uc_node,
        critical=True
    )

    # Add helpful debug info
    evaluator.add_custom_info(
        info={
            "uc_ag_total": count_uc_ag_courses(plan),
            "uc_ag_c_or_better_total": count_uc_ag_c_or_better(plan),
            "uc_ag_before_senior_c_or_better": count_uc_ag_before_senior(plan),
            "uc_ag_gpa": uc_gpa
        },
        info_type="uc_ag_stats"
    )


async def add_ncaa_di_checks(evaluator: Evaluator, parent_node, plan: CoursePlanExtraction, logger: logging.Logger) -> None:
    ncaa_node = evaluator.add_parallel(
        id="NCAA_DI_Requirements",
        desc="All NCAA Division I initial-eligibility core-course requirements in the prompt are satisfied.",
        parent=parent_node,
        critical=True
    )

    # Total 16 core courses
    total_core = sum(1 for _, c in all_courses_with_grade(plan) if c.ncaa_core_area)
    evaluator.add_custom_node(
        result=total_core >= 16,
        id="NCAA_16_Core_Total",
        desc="Includes 16 NCAA core courses total.",
        parent=ncaa_node,
        critical=True
    )

    # Early 10 and 7 of those in EMS
    ten_before, seven_ems, total_early, ems_in_first10 = ncaa_early_10_and_7_ems(plan)
    evaluator.add_custom_node(
        result=ten_before,
        id="NCAA_10_Before_7th_Semester",
        desc="Includes at least 10 NCAA core courses before the start of the 7th semester (senior year).",
        parent=ncaa_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=seven_ems,
        id="NCAA_7_of_10_Are_EMS",
        desc="Of the 10 early NCAA core courses, at least 7 are in English, Math, or Natural/Physical Science.",
        parent=ncaa_node,
        critical=True
    )

    # Core GPA at least 2.3
    ncaa_gpa = compute_ncaa_core_gpa(plan)
    evaluator.add_custom_node(
        result=(ncaa_gpa is not None and ncaa_gpa >= 2.3),
        id="NCAA_Core_GPA_Min_2_3",
        desc="NCAA core-course GPA is at least 2.3.",
        parent=ncaa_node,
        critical=True
    )

    # Subject distribution
    ncaa_subj_node = evaluator.add_parallel(
        id="NCAA_Subject_Distribution",
        desc="NCAA subject distribution requirements are met.",
        parent=ncaa_node,
        critical=True
    )
    ncaa_checks = ncaa_subject_distribution_checks(plan)
    evaluator.add_custom_node(
        result=ncaa_checks["NCAA_English_4"],
        id="NCAA_English_4",
        desc="Includes 4 years of English (core).",
        parent=ncaa_subj_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=ncaa_checks["NCAA_Math_3_AlgIplus"],
        id="NCAA_Math_3_AlgIplus",
        desc="Includes 3 years of mathematics at Algebra I or higher (core).",
        parent=ncaa_subj_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=ncaa_checks["NCAA_Science_2_With_Lab_If_Offered"],
        id="NCAA_Science_2_With_Lab_If_Offered",
        desc="Includes 2 years of natural/physical science, with 1 year of lab if offered by the school (core).",
        parent=ncaa_subj_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=ncaa_checks["NCAA_Social_Science_2"],
        id="NCAA_Social_Science_2",
        desc="Includes 2 years of social science (core).",
        parent=ncaa_subj_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=ncaa_checks["NCAA_Additional_4_EMS"],
        id="NCAA_Additional_4_EMS",
        desc="Includes 4 additional years of English/Math/Science (core).",
        parent=ncaa_subj_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=ncaa_checks["NCAA_Additional_1_From_Allowed_Areas"],
        id="NCAA_Additional_1_From_Allowed_Areas",
        desc="Includes 1 additional year from the allowed areas (English/Math/Science/Social Science/World Language/Comparative Religion/Philosophy).",
        parent=ncaa_subj_node,
        critical=True
    )

    # NCAA-approved list assertion
    evaluator.add_custom_node(
        result=bool(plan.states_ncaa_approved),
        id="NCAA_All_Core_Courses_NCAA_Approved",
        desc="All NCAA core courses listed are on the high school's NCAA-approved core course list (explicitly stated).",
        parent=ncaa_node,
        critical=True
    )

    # Cores within first 8 semesters (grades 9-12 only)
    only_9_12 = is_plan_organized_by_grade(plan)  # if organized strictly by 9-12, we treat as within first 8 semesters
    evaluator.add_custom_node(
        result=only_9_12,
        id="NCAA_Cores_Within_First_8_Semesters",
        desc="NCAA core courses counted toward eligibility are completed within the first 8 semesters from the start of 9th grade.",
        parent=ncaa_node,
        critical=True
    )

    evaluator.add_custom_info(
        info={
            "ncaa_core_total": total_core,
            "ncaa_early_core_total": total_early,
            "ncaa_ems_in_first10": ems_in_first10,
            "ncaa_core_gpa": ncaa_gpa
        },
        info_type="ncaa_core_stats"
    )


async def add_reference_urls_checks(evaluator: Evaluator, parent_node, plan: CoursePlanExtraction) -> None:
    ref_node = evaluator.add_parallel(
        id="Reference_URLs",
        desc="Provides reference URLs to verify the UC A–G and NCAA Division I requirements used.",
        parent=parent_node,
        critical=True
    )

    # UC references provided
    uc_url_leaf = evaluator.add_leaf(
        id="UC_Requirements_URL_Provided",
        desc="Provides at least one reference URL for UC A–G requirements.",
        parent=ref_node,
        critical=True
    )
    uc_claim = "The answer provides at least one valid URL that explains the UC A–G course requirements for California residents."
    await evaluator.verify(
        claim=uc_claim,
        node=uc_url_leaf,
        sources=plan.uc_reference_urls if plan.uc_reference_urls else None,
        additional_instruction="If sources are provided, verify that the page(s) describe UC A–G requirements for California residents."
    )

    # NCAA references provided
    ncaa_url_leaf = evaluator.add_leaf(
        id="NCAA_Requirements_URL_Provided",
        desc="Provides at least one reference URL for NCAA Division I initial-eligibility/core-course requirements.",
        parent=ref_node,
        critical=True
    )
    ncaa_claim = "The answer provides at least one valid URL that explains NCAA Division I initial-eligibility and core-course requirements."
    await evaluator.verify(
        claim=ncaa_claim,
        node=ncaa_url_leaf,
        sources=plan.ncaa_reference_urls if plan.ncaa_reference_urls else None,
        additional_instruction="If sources are provided, verify that the page(s) describe NCAA Division I initial-eligibility and core-course requirements."
    )


# ------------------------------- Main Entry -------------------------------- #
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

    # Extract the structured course plan
    plan: CoursePlanExtraction = await evaluator.extract(
        prompt=prompt_extract_course_plan(),
        template_class=CoursePlanExtraction,
        extraction_name="course_plan_extraction"
    )

    # Build the top-level critical evaluation node to mirror rubric
    top = evaluator.add_parallel(
        id="Course_Plan_Evaluation",
        desc="Evaluation of a four-year (grades 9–12) course plan that satisfies UC A–G (CA resident) and NCAA Division I initial-eligibility requirements, including required presentation and references.",
        parent=root,
        critical=True
    )

    # Subtrees
    await add_plan_presentation_checks(evaluator, top, plan)
    await add_uc_ag_checks(evaluator, top, plan, logger)
    await add_ncaa_di_checks(evaluator, top, plan, logger)
    await add_reference_urls_checks(evaluator, top, plan)

    # Return summary
    return evaluator.get_summary()