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
TASK_ID = "edu_requirements_comparison_ca_oh_magnets"
TASK_DESCRIPTION = (
    "A high school guidance counselor in California is creating an informational resource guide for families relocating "
    "from Ohio to California. The guide must provide accurate comparisons of graduation requirements and program "
    "eligibility standards to help families understand the key differences and ensure their students remain on track "
    "for graduation and college admission.\n\n"
    "Provide a comprehensive comparison that includes:\n\n"
    "1. Ohio State Graduation Requirements: Identify the minimum number of credits required for graduation in Ohio and "
    "specify the mathematics requirement regarding Algebra II completion.\n\n"
    "2. California State Minimum Graduation Requirements: Identify the minimum course requirements for English, "
    "Mathematics (including specific Algebra requirements), and Science for California state graduation.\n\n"
    "3. California A-G Requirements: Specify the total number of college-preparatory courses required for UC/CSU "
    "admission eligibility, the minimum grade required, how many must be completed before the final year, and the "
    "specific requirements for English and Mathematics.\n\n"
    "4. Basic Magnet Program Eligibility: Identify the minimum GPA requirement in core academic subjects and the "
    "attendance requirements (maximum unexcused absences) for basic/random selection magnet programs.\n\n"
    "5. Enhanced Magnet Program Eligibility: Identify the minimum GPA requirement in core academic subjects for "
    "enhanced/competitive magnet programs (such as AP Capstone, Cambridge International, IB, or similar advanced "
    "programs) and any specific prerequisite course requirements for grade-level applicants.\n\n"
    "Each requirement identified must be supported by a valid reference URL from an official educational authority or "
    "school district website."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OhioRequirements(BaseModel):
    min_credits: Optional[str] = None
    math_algebra_ii: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CAStateRequirements(BaseModel):
    english_courses: Optional[str] = None
    math_courses: Optional[str] = None
    math_includes_alg1: Optional[str] = None
    science_courses: Optional[str] = None
    science_includes_bio_phys: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AGRequirements(BaseModel):
    total_yearlong_courses: Optional[str] = None
    min_grade: Optional[str] = None
    eleven_before_final_year: Optional[str] = None
    english_years: Optional[str] = None
    math_required_years: Optional[str] = None
    math_recommended_years: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BasicMagnetRequirements(BaseModel):
    min_gpa: Optional[str] = None
    gpa_time_window: Optional[str] = None
    max_unexcused_prev_year: Optional[str] = None
    max_unexcused_current_first_semester: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EnhancedMagnetRequirements(BaseModel):
    min_gpa: Optional[str] = None
    gpa_time_window: Optional[str] = None
    prereq_alg1_9th: Optional[str] = None
    prereq_geometry_10th: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RequirementsExtraction(BaseModel):
    ohio: Optional[OhioRequirements] = None
    ca_state: Optional[CAStateRequirements] = None
    ag: Optional[AGRequirements] = None
    basic_magnet: Optional[BasicMagnetRequirements] = None
    enhanced_magnet: Optional[EnhancedMagnetRequirements] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    You will extract a structured summary of the requirements explicitly stated in the answer, along with the URLs the answer cites for each section. Do not infer or invent anything not present in the answer. Use exact values/wording as written (e.g., “20 credits,” “3 courses,” “4 years,” “C or better,” “Algebra II,” “Algebra I,” “2.0 GPA,” specific unexcused absences thresholds, etc.). Return null for any item not explicitly stated in the answer.

    Extract the following sections:

    1) ohio:
       - min_credits: The minimum number of credits/units required for Ohio high school graduation (e.g., "20 credits", "20 units").
       - math_algebra_ii: The math requirement statement regarding Algebra II or equivalent (e.g., "completion through Algebra II or equivalent").
       - sources: All URLs the answer cites for these Ohio items.

    2) ca_state:
       - english_courses: The number of courses/years in English required by California’s state minimum graduation requirements (e.g., "3 courses", "3 years").
       - math_courses: The number of courses/years in Mathematics required by California’s state minimum (e.g., "2 courses").
       - math_includes_alg1: The Algebra I inclusion statement if mentioned (e.g., "one year of Algebra I", "includes Algebra I").
       - science_courses: The number of courses/years in Science required by California’s state minimum (e.g., "2 courses").
       - science_includes_bio_phys: The biological and physical sciences inclusion statement if mentioned.
       - sources: All URLs the answer cites for these California state minimum items.

    3) ag:
       - total_yearlong_courses: The total number of A–G yearlong college-prep courses required for UC/CSU eligibility (e.g., "15").
       - min_grade: The minimum letter grade required in A–G courses (e.g., "C or better", "C–").
       - eleven_before_final_year: The number of A–G courses to complete before the last year (e.g., "11 of the 15").
       - english_years: The number of years of English required (e.g., "4 years").
       - math_required_years: The number of years of Math required (e.g., "3 years").
       - math_recommended_years: The recommended years of Math if stated (e.g., "4 years recommended").
       - sources: All URLs the answer cites for these A–G items.

    4) basic_magnet:
       - min_gpa: Minimum GPA in core academic subjects for basic/random selection magnet eligibility (e.g., "2.0").
       - gpa_time_window: If stated, the GPA time window definition (e.g., "previous year plus first grading period of current year combined").
       - max_unexcused_prev_year: Maximum unexcused absences allowed for the previous school year (e.g., "10").
       - max_unexcused_current_first_semester: Maximum unexcused absences allowed for the first semester of the current year (e.g., "5").
       - sources: All URLs the answer cites for basic magnet eligibility.

    5) enhanced_magnet:
       - min_gpa: Minimum GPA in core academic subjects for enhanced/competitive magnet programs (e.g., "2.5").
       - gpa_time_window: If stated, the GPA time window definition for enhanced magnets.
       - prereq_alg1_9th: If stated, Algebra I completion requirement for 9th grade applicants.
       - prereq_geometry_10th: If stated, Geometry completion requirement for 10th grade applicants.
       - sources: All URLs the answer cites for enhanced magnet eligibility.

    URL extraction rules:
    - Extract only URLs explicitly presented in the answer (plain links, markdown links, or recognizable in text).
    - Include all relevant URLs for the given section. If none are provided, return an empty list for that section’s sources.
    - Do not infer or construct unseen URLs. Prepend http:// only if the protocol is entirely missing.

    Return a single JSON object following the specified schema fields. Do not include additional commentary.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _official_url_instruction(section_hint: str) -> str:
    return (
        "Treat a URL as 'official' only if it is from a governmental or educational authority, including:\n"
        "- .gov or .edu domains (e.g., state department of education, UC/CSU official admissions sites), or\n"
        "- an official public school district website (e.g., *.k12.ca.us, lausd.org, sfusd.edu, etc.).\n"
        "Do not consider blogs, commercial test-prep, or unofficial summaries as official.\n"
        f"Evaluate the URLs specifically for: {section_hint}.\n"
    )


def _sources_present(evaluator: Evaluator, parent, urls: List[str], id_base: str, desc: str, critical: bool = True):
    return evaluator.add_custom_node(
        result=bool(urls),
        id=id_base,
        desc=desc,
        parent=parent,
        critical=critical
    )


# --------------------------------------------------------------------------- #
# Section verifications                                                       #
# --------------------------------------------------------------------------- #
async def verify_ohio_section(evaluator: Evaluator, parent, data: Optional[OhioRequirements]) -> None:
    node = evaluator.add_parallel(
        id="Ohio_State_Graduation_Requirements",
        desc="Ohio state-level high school graduation requirements.",
        parent=parent,
        critical=True
    )

    urls = data.sources if data and data.sources else []
    _sources_present(
        evaluator,
        node,
        urls,
        id_base="Ohio_sources_provided",
        desc="Ohio: at least one official reference URL is provided",
        critical=True
    )

    # Ohio_Minimum_Credits_20
    if data and data.min_credits:
        leaf = evaluator.add_leaf(
            id="Ohio_Minimum_Credits_20",
            desc="States that Ohio requires a minimum of 20 credits for high school graduation.",
            parent=node,
            critical=True
        )
        claim = f"Ohio requires a minimum of {data.min_credits} for high school graduation."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=(
                "Verify using Ohio Department of Education or other official state sources. "
                "Allow synonymous wording like 'units' vs 'credits' if clearly equivalent and state-level."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Ohio_Minimum_Credits_20",
            desc="States that Ohio requires a minimum of 20 credits for high school graduation. (Missing in answer)",
            parent=node,
            critical=True
        )

    # Ohio_Math_Through_Algebra_II
    if data and data.math_algebra_ii:
        leaf = evaluator.add_leaf(
            id="Ohio_Math_Through_Algebra_II",
            desc="States that Ohio mathematics requirements include completion through Algebra II (or equivalent).",
            parent=node,
            critical=True
        )
        claim = f"Ohio's high school math requirement includes completion through {data.math_algebra_ii}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=(
                "Confirm the Algebra II (or equivalent) requirement from official Ohio sources. "
                "Allow equivalent phrasing like 'Algebra 2' or 'advanced algebra/Algebra II or equivalent'."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Ohio_Math_Through_Algebra_II",
            desc="States that Ohio mathematics requirements include completion through Algebra II (or equivalent). (Missing in answer)",
            parent=node,
            critical=True
        )


async def verify_ca_state_min_section(evaluator: Evaluator, parent, data: Optional[CAStateRequirements]) -> None:
    node = evaluator.add_parallel(
        id="California_State_Minimum_Graduation_Requirements",
        desc="California state minimum graduation course requirements for English, Mathematics (including Algebra), and Science.",
        parent=parent,
        critical=True
    )

    urls = data.sources if data and data.sources else []
    _sources_present(
        evaluator,
        node,
        urls,
        id_base="CA_state_sources_provided",
        desc="California state minimum: at least one official reference URL is provided",
        critical=True
    )

    # CA_English_3_Courses
    if data and data.english_courses:
        leaf = evaluator.add_leaf(
            id="CA_English_3_Courses",
            desc="States that California state minimum graduation requires three courses in English.",
            parent=node,
            critical=True
        )
        claim = f"California's state minimum graduation requires {data.english_courses} in English."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction="Validate against official California Department of Education or district policy pages. Treat 'courses'/'years' as equivalent when clearly intended.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="CA_English_3_Courses",
            desc="States that California state minimum graduation requires three courses in English. (Missing in answer)",
            parent=node,
            critical=True
        )

    # CA_Math_2_Courses
    if data and data.math_courses:
        leaf = evaluator.add_leaf(
            id="CA_Math_2_Courses",
            desc="States that California state minimum graduation requires two courses in Mathematics.",
            parent=node,
            critical=True
        )
        claim = f"California's state minimum graduation requires {data.math_courses} in Mathematics."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction="Validate on official California sources. Accept minor phrasing variants.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="CA_Math_2_Courses",
            desc="States that California state minimum graduation requires two courses in Mathematics. (Missing in answer)",
            parent=node,
            critical=True
        )

    # CA_Math_Includes_Algebra_I_1_Year
    if data and data.math_includes_alg1:
        leaf = evaluator.add_leaf(
            id="CA_Math_Includes_Algebra_I_1_Year",
            desc="States that California state minimum Mathematics includes one year of Algebra I.",
            parent=node,
            critical=True
        )
        claim = f"California state minimum Mathematics includes {data.math_includes_alg1}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction="Confirm that Algebra I is included in California's state minimum Math requirements; allow phrasing like 'includes Algebra I' or 'one year Algebra I'.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="CA_Math_Includes_Algebra_I_1_Year",
            desc="States that California state minimum Mathematics includes one year of Algebra I. (Missing in answer)",
            parent=node,
            critical=True
        )

    # CA_Science_2_Courses
    if data and data.science_courses:
        leaf = evaluator.add_leaf(
            id="CA_Science_2_Courses",
            desc="States that California state minimum graduation requires two courses in Science.",
            parent=node,
            critical=True
        )
        claim = f"California's state minimum graduation requires {data.science_courses} in Science."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction="Confirm via official CDE or district pages; allow 'courses'/'years' equivalence if clearly intended.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="CA_Science_2_Courses",
            desc="States that California state minimum graduation requires two courses in Science. (Missing in answer)",
            parent=node,
            critical=True
        )

    # CA_Science_Includes_Biological_And_Physical
    if data and data.science_includes_bio_phys:
        leaf = evaluator.add_leaf(
            id="CA_Science_Includes_Biological_And_Physical",
            desc="States that the California state minimum Science requirement includes biological and physical sciences.",
            parent=node,
            critical=True
        )
        claim = f"California state minimum Science includes {data.science_includes_bio_phys}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction="Verify the inclusion of both biological and physical sciences in the state minimum Science requirement.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="CA_Science_Includes_Biological_And_Physical",
            desc="States that the California state minimum Science requirement includes biological and physical sciences. (Missing in answer)",
            parent=node,
            critical=True
        )


async def verify_ag_section(evaluator: Evaluator, parent, data: Optional[AGRequirements]) -> None:
    node = evaluator.add_parallel(
        id="California_AG_Requirements",
        desc="UC/CSU A–G admission eligibility requirements.",
        parent=parent,
        critical=True
    )

    urls = data.sources if data and data.sources else []
    _sources_present(
        evaluator,
        node,
        urls,
        id_base="AG_sources_provided",
        desc="A–G: at least one official reference URL is provided",
        critical=True
    )

    # AG_Total_15_Yearlong_Courses
    if data and data.total_yearlong_courses:
        leaf = evaluator.add_leaf(
            id="AG_Total_15_Yearlong_Courses",
            desc="States that A–G requires 15 yearlong college-preparatory courses.",
            parent=node,
            critical=True
        )
        claim = f"A–G requires {data.total_yearlong_courses} yearlong college-preparatory courses."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction="Use UC/CSU official admissions pages; allow phrasing like '15 units (yearlong courses)'.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="AG_Total_15_Yearlong_Courses",
            desc="States that A–G requires 15 yearlong college-preparatory courses. (Missing in answer)",
            parent=node,
            critical=True
        )

    # AG_Minimum_Grade_C_or_Better
    if data and data.min_grade:
        leaf = evaluator.add_leaf(
            id="AG_Minimum_Grade_C_or_Better",
            desc="States that A–G requires a minimum letter grade of C (or better).",
            parent=node,
            critical=True
        )
        claim = f"A–G courses must be completed with a minimum grade of {data.min_grade}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction="Validate on official UC/CSU guidance (e.g., 'grade of C or better').",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="AG_Minimum_Grade_C_or_Better",
            desc="States that A–G requires a minimum letter grade of C (or better). (Missing in answer)",
            parent=node,
            critical=True
        )

    # AG_11_Completed_Before_Final_Year
    if data and data.eleven_before_final_year:
        leaf = evaluator.add_leaf(
            id="AG_11_Completed_Before_Final_Year",
            desc="States that at least 11 of the 15 A–G courses must be completed prior to the last year of high school.",
            parent=node,
            critical=True
        )
        claim = f"At least {data.eleven_before_final_year} of the A–G courses must be completed prior to the final year of high school."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction="Confirm UC/CSU requirement for completing 11 of 15 A–G courses before the last year.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="AG_11_Completed_Before_Final_Year",
            desc="States that at least 11 of the 15 A–G courses must be completed prior to the last year of high school. (Missing in answer)",
            parent=node,
            critical=True
        )

    # AG_English_4_Years
    if data and data.english_years:
        leaf = evaluator.add_leaf(
            id="AG_English_4_Years",
            desc="States that A–G requires 4 years of English.",
            parent=node,
            critical=True
        )
        claim = f"A–G requires {data.english_years} of English."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction="Validate against UC/CSU official A–G area 'b' (English) requirement.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="AG_English_4_Years",
            desc="States that A–G requires 4 years of English. (Missing in answer)",
            parent=node,
            critical=True
        )

    # AG_Math_3_Years_4_Recommended
    if data and data.math_required_years and data.math_recommended_years:
        leaf = evaluator.add_leaf(
            id="AG_Math_3_Years_4_Recommended",
            desc="States that A–G requires 3 years of Mathematics (with 4 years recommended).",
            parent=node,
            critical=True
        )
        claim = (
            f"A–G requires {data.math_required_years} of Mathematics, and {data.math_recommended_years} is recommended."
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction="Validate on UC/CSU A–G area 'c' (Mathematics): 3 years required, 4 years recommended.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="AG_Math_3_Years_4_Recommended",
            desc="States that A–G requires 3 years of Mathematics (with 4 years recommended). (Missing in answer)",
            parent=node,
            critical=True
        )


async def verify_basic_magnet_section(evaluator: Evaluator, parent, data: Optional[BasicMagnetRequirements]) -> None:
    node = evaluator.add_parallel(
        id="Basic_Magnet_Program_Eligibility",
        desc="Eligibility requirements for basic/random selection magnet programs.",
        parent=parent,
        critical=True
    )

    urls = data.sources if data and data.sources else []
    _sources_present(
        evaluator,
        node,
        urls,
        id_base="Basic_magnet_sources_provided",
        desc="Basic magnet: at least one official reference URL is provided",
        critical=True
    )

    # Basic_Magnet_Min_GPA_2_0
    if data and data.min_gpa:
        leaf = evaluator.add_leaf(
            id="Basic_Magnet_Min_GPA_2_0",
            desc="States that basic magnet programs require a minimum 2.0 GPA in core academic subjects.",
            parent=node,
            critical=True
        )
        claim = f"Basic/random selection magnet programs require at least a {data.min_gpa} GPA in core academic subjects."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction="Use official district/school magnet policy pages. Core subjects typically include English, Math, Science, Social Studies.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Basic_Magnet_Min_GPA_2_0",
            desc="States that basic magnet programs require a minimum 2.0 GPA in core academic subjects. (Missing in answer)",
            parent=node,
            critical=True
        )

    # Basic_Magnet_GPA_Time_Window_Definition
    if data and data.gpa_time_window:
        leaf = evaluator.add_leaf(
            id="Basic_Magnet_GPA_Time_Window_Definition",
            desc="States that the basic magnet GPA is based on the previous year plus the first grading period of the current year combined (as specified in constraints).",
            parent=node,
            critical=True
        )
        claim = f"For basic magnet eligibility, GPA is based on {data.gpa_time_window}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction="Confirm that the GPA window combines the previous school year with the current year's first grading period.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Basic_Magnet_GPA_Time_Window_Definition",
            desc="States that the basic magnet GPA time window is previous year + first grading period combined. (Missing in answer)",
            parent=node,
            critical=True
        )

    # Basic_Magnet_Max_Unexcused_Absences_Prev_Year_10
    if data and data.max_unexcused_prev_year:
        leaf = evaluator.add_leaf(
            id="Basic_Magnet_Max_Unexcused_Absences_Prev_Year_10",
            desc="States that basic magnet eligibility requires no more than 10 unexcused absences for the previous year.",
            parent=node,
            critical=True
        )
        claim = f"Basic magnet eligibility allows no more than {data.max_unexcused_prev_year} unexcused absences for the previous year."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction="Verify attendance thresholds from official district magnet guidelines.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Basic_Magnet_Max_Unexcused_Absences_Prev_Year_10",
            desc="States that basic magnet eligibility requires no more than 10 unexcused absences for the previous year. (Missing in answer)",
            parent=node,
            critical=True
        )

    # Basic_Magnet_Max_Unexcused_Absences_Current_First_Semester_5
    if data and data.max_unexcused_current_first_semester:
        leaf = evaluator.add_leaf(
            id="Basic_Magnet_Max_Unexcused_Absences_Current_First_Semester_5",
            desc="States that basic magnet eligibility requires no more than 5 unexcused absences for the first semester of the current year.",
            parent=node,
            critical=True
        )
        claim = f"Basic magnet eligibility allows no more than {data.max_unexcused_current_first_semester} unexcused absences in the first semester of the current year."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction="Verify attendance thresholds using official district/school magnet documentation.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Basic_Magnet_Max_Unexcused_Absences_Current_First_Semester_5",
            desc="States that basic magnet eligibility requires no more than 5 unexcused absences for the first semester of the current year. (Missing in answer)",
            parent=node,
            critical=True
        )


async def verify_enhanced_magnet_section(evaluator: Evaluator, parent, data: Optional[EnhancedMagnetRequirements]) -> None:
    node = evaluator.add_parallel(
        id="Enhanced_Magnet_Program_Eligibility",
        desc="Eligibility requirements for enhanced/competitive magnet programs (e.g., AP Capstone, Cambridge International, IB, or similar advanced programs).",
        parent=parent,
        critical=True
    )

    urls = data.sources if data and data.sources else []
    _sources_present(
        evaluator,
        node,
        urls,
        id_base="Enhanced_magnet_sources_provided",
        desc="Enhanced/competitive magnet: at least one official reference URL is provided",
        critical=True
    )

    # Enhanced_Magnet_Min_GPA_2_5
    if data and data.min_gpa:
        leaf = evaluator.add_leaf(
            id="Enhanced_Magnet_Min_GPA_2_5",
            desc="States that enhanced/competitive magnet programs require a minimum 2.5 GPA in core academic subjects.",
            parent=node,
            critical=True
        )
        claim = f"Enhanced/competitive magnet programs require at least a {data.min_gpa} GPA in core academic subjects."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction="Use official district/school magnet criteria pages; core subjects typically include English, Math, Science, and Social Studies.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Enhanced_Magnet_Min_GPA_2_5",
            desc="States that enhanced/competitive magnet programs require a minimum 2.5 GPA in core academic subjects. (Missing in answer)",
            parent=node,
            critical=True
        )

    # Enhanced_Magnet_GPA_Time_Window_Definition
    if data and data.gpa_time_window:
        leaf = evaluator.add_leaf(
            id="Enhanced_Magnet_GPA_Time_Window_Definition",
            desc="States that the enhanced/competitive magnet GPA is based on the previous year plus the first grading period of the current year combined (as specified in constraints).",
            parent=node,
            critical=True
        )
        claim = f"For enhanced/competitive magnet eligibility, GPA is based on {data.gpa_time_window}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction="Confirm that the GPA window is defined as previous year + first grading period of the current year.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Enhanced_Magnet_GPA_Time_Window_Definition",
            desc="States that the enhanced/competitive magnet GPA time window is previous year + first grading period combined. (Missing in answer)",
            parent=node,
            critical=True
        )

    # Enhanced_Magnet_Prereq_Algebra_I_For_9th
    if data and data.prereq_alg1_9th:
        leaf = evaluator.add_leaf(
            id="Enhanced_Magnet_Prereq_Algebra_I_For_9th",
            desc="States that select enhanced programs may require Algebra I completion for 9th grade applicants before the start of the school year.",
            parent=node,
            critical=True
        )
        claim = f"Select enhanced magnet programs require {data.prereq_alg1_9th} for 9th grade applicants before the start of the school year."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction="Verify Algebra I prerequisite expectations for 9th grade entry from official program pages.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Enhanced_Magnet_Prereq_Algebra_I_For_9th",
            desc="States that select enhanced programs may require Algebra I completion for 9th grade applicants. (Missing in answer)",
            parent=node,
            critical=True
        )

    # Enhanced_Magnet_Prereq_Geometry_For_10th
    if data and data.prereq_geometry_10th:
        leaf = evaluator.add_leaf(
            id="Enhanced_Magnet_Prereq_Geometry_For_10th",
            desc="States that select enhanced programs may require Geometry completion for 10th grade applicants before the start of the school year.",
            parent=node,
            critical=True
        )
        claim = f"Select enhanced magnet programs require {data.prereq_geometry_10th} for 10th grade applicants before the start of the school year."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction="Verify Geometry prerequisite expectations for 10th grade entry from official program pages.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Enhanced_Magnet_Prereq_Geometry_For_10th",
            desc="States that select enhanced programs may require Geometry completion for 10th grade applicants. (Missing in answer)",
            parent=node,
            critical=True
        )


async def verify_official_urls_section(
    evaluator: Evaluator,
    parent,
    extracted: RequirementsExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Official_Reference_URLs",
        desc="Provides valid reference URL(s) from official educational authority or school district websites supporting the stated requirements, broken out by section for atomic evaluation.",
        parent=parent,
        critical=True
    )

    # Ohio
    ohio_urls = extracted.ohio.sources if extracted.ohio and extracted.ohio.sources else []
    if ohio_urls:
        leaf = evaluator.add_leaf(
            id="Official_URLs_Ohio_Requirements",
            desc="Provides official reference URL(s) supporting the Ohio graduation requirements stated (minimum credits and Algebra II math requirement).",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim="These URLs are from official educational authorities or school district websites.",
            node=leaf,
            sources=ohio_urls,
            additional_instruction=_official_url_instruction("Ohio graduation requirements (state-level)"),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Official_URLs_Ohio_Requirements",
            desc="Provides official reference URL(s) supporting the Ohio graduation requirements. (Missing URLs in answer)",
            parent=node,
            critical=True
        )

    # California state minimum
    ca_state_urls = extracted.ca_state.sources if extracted.ca_state and extracted.ca_state.sources else []
    if ca_state_urls:
        leaf = evaluator.add_leaf(
            id="Official_URLs_CA_State_Minimum_Requirements",
            desc="Provides official reference URL(s) supporting the California state minimum graduation requirements stated (English, Math incl. Algebra I, Science incl. biological & physical).",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim="These URLs are from official educational authorities or school district websites.",
            node=leaf,
            sources=ca_state_urls,
            additional_instruction=_official_url_instruction("California state minimum high school graduation requirements"),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Official_URLs_CA_State_Minimum_Requirements",
            desc="Provides official reference URL(s) supporting California state minimum graduation requirements. (Missing URLs in answer)",
            parent=node,
            critical=True
        )

    # A–G
    ag_urls = extracted.ag.sources if extracted.ag and extracted.ag.sources else []
    if ag_urls:
        leaf = evaluator.add_leaf(
            id="Official_URLs_CA_AG_Requirements",
            desc="Provides official reference URL(s) supporting the California A–G requirements stated (15 courses, C-or-better, 11-before-final-year, English, Math).",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim="These URLs are from official educational authorities or school district websites.",
            node=leaf,
            sources=ag_urls,
            additional_instruction=_official_url_instruction("UC/CSU A–G eligibility requirements (UC/CSU official admissions pages)"),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Official_URLs_CA_AG_Requirements",
            desc="Provides official reference URL(s) supporting A–G requirements. (Missing URLs in answer)",
            parent=node,
            critical=True
        )

    # Basic magnet
    basic_urls = extracted.basic_magnet.sources if extracted.basic_magnet and extracted.basic_magnet.sources else []
    if basic_urls:
        leaf = evaluator.add_leaf(
            id="Official_URLs_Basic_Magnet_Eligibility",
            desc="Provides official reference URL(s) supporting the basic magnet program eligibility requirements stated (GPA threshold and attendance thresholds, including the stated GPA time window definition if included).",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim="These URLs are from official educational authorities or school district websites.",
            node=leaf,
            sources=basic_urls,
            additional_instruction=_official_url_instruction("basic/random selection magnet eligibility"),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Official_URLs_Basic_Magnet_Eligibility",
            desc="Provides official reference URL(s) supporting basic magnet eligibility. (Missing URLs in answer)",
            parent=node,
            critical=True
        )

    # Enhanced magnet
    enhanced_urls = extracted.enhanced_magnet.sources if extracted.enhanced_magnet and extracted.enhanced_magnet.sources else []
    if enhanced_urls:
        leaf = evaluator.add_leaf(
            id="Official_URLs_Enhanced_Magnet_Eligibility",
            desc="Provides official reference URL(s) supporting the enhanced/competitive magnet program eligibility requirements stated (GPA threshold, the stated GPA time window definition if included, and prerequisite course requirements if included).",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim="These URLs are from official educational authorities or school district websites.",
            node=leaf,
            sources=enhanced_urls,
            additional_instruction=_official_url_instruction("enhanced/competitive magnet eligibility (e.g., AP Capstone, IB, Cambridge)"),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Official_URLs_Enhanced_Magnet_Eligibility",
            desc="Provides official reference URL(s) supporting enhanced/competitive magnet eligibility. (Missing URLs in answer)",
            parent=node,
            critical=True
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

    # 1) Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=RequirementsExtraction,
        extraction_name="requirements_extraction",
    )

    # 2) Build main comparison node (critical, parallel aggregation)
    comparison_node = evaluator.add_parallel(
        id="Educational_Requirements_Comparison",
        desc="Comparison guide covering Ohio graduation requirements, California graduation minimums, California A–G requirements, and magnet program eligibility requirements, with official citations.",
        parent=root,
        critical=True
    )

    # 3) Verify each section
    await verify_ohio_section(evaluator, comparison_node, extracted.ohio)
    await verify_ca_state_min_section(evaluator, comparison_node, extracted.ca_state)
    await verify_ag_section(evaluator, comparison_node, extracted.ag)
    await verify_basic_magnet_section(evaluator, comparison_node, extracted.basic_magnet)
    await verify_enhanced_magnet_section(evaluator, comparison_node, extracted.enhanced_magnet)

    # 4) Verify presence and officialness of URLs by section (atomic checks)
    await verify_official_urls_section(evaluator, comparison_node, extracted)

    # Optional: Record simple diagnostics about URL counts
    evaluator.add_custom_info(
        info={
            "ohio_urls": len(extracted.ohio.sources) if extracted.ohio and extracted.ohio.sources else 0,
            "ca_state_urls": len(extracted.ca_state.sources) if extracted.ca_state and extracted.ca_state.sources else 0,
            "ag_urls": len(extracted.ag.sources) if extracted.ag and extracted.ag.sources else 0,
            "basic_magnet_urls": len(extracted.basic_magnet.sources) if extracted.basic_magnet and extracted.basic_magnet.sources else 0,
            "enhanced_magnet_urls": len(extracted.enhanced_magnet.sources) if extracted.enhanced_magnet and extracted.enhanced_magnet.sources else 0,
        },
        info_type="url_stats"
    )

    # 5) Return evaluation summary
    return evaluator.get_summary()