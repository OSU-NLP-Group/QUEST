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
TASK_ID = "ncaa_d1_eligibility_requirements"
TASK_DESCRIPTION = (
    "I am a high school junior planning to compete in NCAA Division I athletics while attending college. "
    "I need to understand all the academic eligibility requirements I must meet. Please provide a comprehensive overview that includes: "
    "(1) the complete breakdown of core course requirements by subject area and the total number needed, "
    "(2) the timeline requirements for when these courses must be completed, "
    "(3) the GPA requirements, "
    "(4) the standardized test score requirements, and "
    "(5) key rules about transfer eligibility and degree completion that I should know. "
    "Be specific about numbers and deadlines."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class NCAARequirementsExtraction(BaseModel):
    # Core course counts and breakdown (as stated in the answer)
    total_core_courses: Optional[str] = None
    english_years: Optional[str] = None
    math_years: Optional[str] = None
    mentions_algebra1_or_higher: Optional[bool] = None
    science_years: Optional[str] = None
    mentions_lab_if_offered: Optional[bool] = None
    social_science_years: Optional[str] = None
    extra_eng_math_sci_year: Optional[str] = None
    additional_four_core_years: Optional[str] = None
    mentions_allowed_categories: Optional[bool] = None  # e.g., foreign language, philosophy, non-doctrinal religion

    # Timeline and progress rules
    eight_semester_timeframe: Optional[str] = None
    ten_before_7th: Optional[str] = None
    seven_of_ten_in_core_subjects: Optional[bool] = None  # True if answer specifies 7 of the 10 are in Eng/Math/Sci

    # GPA and test score rules
    min_competition_gpa: Optional[str] = None
    sliding_scale_mentioned: Optional[bool] = None

    # Transfer/clock/degree info
    five_year_eligibility_clock: Optional[str] = None
    transfer_academic_residence: Optional[str] = None
    bachelors_degree_credits: Optional[str] = None

    # Sources: general and topic-specific URLs explicitly mentioned in the answer
    sources_general: List[str] = Field(default_factory=list)
    sources_total_core: List[str] = Field(default_factory=list)
    sources_english: List[str] = Field(default_factory=list)
    sources_math: List[str] = Field(default_factory=list)
    sources_science: List[str] = Field(default_factory=list)
    sources_social: List[str] = Field(default_factory=list)
    sources_extra_ems: List[str] = Field(default_factory=list)
    sources_additional_four: List[str] = Field(default_factory=list)
    sources_eight_sem: List[str] = Field(default_factory=list)
    sources_gpa: List[str] = Field(default_factory=list)
    sources_tests: List[str] = Field(default_factory=list)
    sources_ten_before7: List[str] = Field(default_factory=list)
    sources_seven_of_ten: List[str] = Field(default_factory=list)
    sources_five_year_clock: List[str] = Field(default_factory=list)
    sources_transfer: List[str] = Field(default_factory=list)
    sources_120_credits: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ncaa_requirements() -> str:
    return """
Extract the NCAA Division I initial eligibility academic rules as explicitly stated in the answer, along with any URLs the answer cites. Do NOT infer facts that are not present in the answer.

For each field below, return exactly what the answer claims (verbatim if possible). If the answer does not state the item, set it to null (or false for booleans). Also extract URLs per topic if the answer associates specific sources with that topic. Additionally, extract a general list of all URLs cited anywhere in the answer.

Fields to extract:
1) total_core_courses: the total number of required core courses (e.g., "16").
2) english_years: number of required years of English (e.g., "4").
3) math_years: number of required years of mathematics (e.g., "3").
4) mentions_algebra1_or_higher: boolean; true only if the answer explicitly mentions that math must be at Algebra I level or higher.
5) science_years: number of required years of natural/physical science (e.g., "2").
6) mentions_lab_if_offered: boolean; true only if the answer explicitly states that one year must be a lab science if offered by the high school.
7) social_science_years: number of required years of social science (e.g., "2").
8) extra_eng_math_sci_year: the additional required year in English, math, or natural/physical science (e.g., "1").
9) additional_four_core_years: the number of additional core years from approved categories (e.g., "4").
10) mentions_allowed_categories: boolean; true only if the answer explicitly mentions categories such as foreign language, philosophy, or non-doctrinal religion.
11) eight_semester_timeframe: number of semesters by which the 16 core courses must be completed (e.g., "8").
12) min_competition_gpa: the minimum core-course GPA to compete in the first year for students enrolling August 2016 or later (e.g., "2.3").
13) sliding_scale_mentioned: boolean; true only if the answer explicitly states that SAT or ACT scores must meet a sliding scale based on GPA.
14) ten_before_7th: how many of the 16 core courses must be completed before the start of the seventh semester (e.g., "10").
15) seven_of_ten_in_core_subjects: boolean; true only if the answer explicitly states that at least 7 of those early courses are in English, math, and science.
16) five_year_eligibility_clock: the number of calendar years to complete four seasons of competition (e.g., "5").
17) transfer_academic_residence: text describing whether a transfer from one four-year institution typically must complete one academic year in residence before competing.
18) bachelors_degree_credits: number of credit hours typically required for a bachelor's degree (e.g., "120").

Also extract the following URL lists:
- sources_general: all URLs referenced anywhere in the answer.
- sources_total_core: URLs specifically cited for total core courses.
- sources_english: URLs specifically cited for English requirements.
- sources_math: URLs specifically cited for math requirements.
- sources_science: URLs specifically cited for science requirements.
- sources_social: URLs specifically cited for social science requirements.
- sources_extra_ems: URLs specifically cited for the extra English/math/science year.
- sources_additional_four: URLs specifically cited for the additional four core years/categories.
- sources_eight_sem: URLs specifically cited for the eight-semester timeframe.
- sources_gpa: URLs specifically cited for the minimum GPA.
- sources_tests: URLs specifically cited for SAT/ACT sliding scale.
- sources_ten_before7: URLs specifically cited for the "10 before seventh semester" rule.
- sources_seven_of_ten: URLs specifically cited for the "7 of 10 in English/Math/Science" rule.
- sources_five_year_clock: URLs specifically cited for the five-year eligibility clock.
- sources_transfer: URLs specifically cited for transfer academic residence rules.
- sources_120_credits: URLs specifically cited for the bachelor's 120 credit hours requirement.

URL extraction rules:
- Only include actual URLs explicitly present in the answer (including markdown links). Do not infer URLs.
- If no URLs are provided for a topic, return an empty list for that topic.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _val_or_missing(value: Optional[str], placeholder: str = "[missing]") -> str:
    if value is None:
        return placeholder
    v = value.strip()
    return v if v else placeholder


def _merge_sources(*lists: List[str]) -> List[str]:
    dedup = []
    seen = set()
    for lst in lists:
        for url in lst:
            if not isinstance(url, str):
                continue
            u = url.strip()
            if not u:
                continue
            # Basic filter to avoid clearly invalid tokens
            if not (u.startswith("http://") or u.startswith("https://")):
                continue
            if u not in seen:
                seen.add(u)
                dedup.append(u)
    return dedup


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_ncaa_tree(evaluator: Evaluator, root, extr: NCAARequirementsExtraction) -> None:
    claims_and_sources = []

    # Total core courses
    node_total = evaluator.add_leaf(
        id="Total_Core_Course_Count",
        desc="States that exactly 16 core courses are required",
        parent=root,
        critical=True
    )
    total_val = _val_or_missing(extr.total_core_courses)
    claim_total = f"NCAA Division I initial eligibility requires exactly {total_val} core courses."
    src_total = _merge_sources(extr.sources_total_core, extr.sources_general)
    addins_total = (
        "Only mark as supported if the evidence explicitly states that 16 total core courses are required. "
        "If the claim's number is not 16 or is missing/ambiguous, treat it as not supported."
    )
    claims_and_sources.append((claim_total, src_total if src_total else None, node_total, addins_total))

    # English years
    node_eng = evaluator.add_leaf(
        id="English_Years",
        desc="States that 4 years of English are required",
        parent=root,
        critical=True
    )
    eng_val = _val_or_missing(extr.english_years)
    claim_eng = f"{eng_val} years of English are required for NCAA Division I initial eligibility."
    src_eng = _merge_sources(extr.sources_english, extr.sources_general)
    addins_eng = (
        "Only mark as supported if the evidence explicitly indicates that 4 years of English are required. "
        "If the claim does not say '4', do not support."
    )
    claims_and_sources.append((claim_eng, src_eng if src_eng else None, node_eng, addins_eng))

    # Math years with Algebra I or higher qualifier
    node_math = evaluator.add_leaf(
        id="Math_Years",
        desc="States that 3 years of mathematics at Algebra 1 level or higher are required",
        parent=root,
        critical=True
    )
    math_val = _val_or_missing(extr.math_years)
    algebra_phrase = " at Algebra I level or higher" if extr.mentions_algebra1_or_higher else ""
    claim_math = f"{math_val} years of mathematics{algebra_phrase} are required for NCAA Division I initial eligibility."
    src_math = _merge_sources(extr.sources_math, extr.sources_general)
    addins_math = (
        "Only mark as supported if the evidence explicitly indicates both that: "
        "(a) 3 years of mathematics are required AND "
        "(b) those years must be at Algebra I level or higher. "
        "If the claim lacks the 'Algebra I or higher' qualifier or the number is not 3, do not support."
    )
    claims_and_sources.append((claim_math, src_math if src_math else None, node_math, addins_math))

    # Science years and lab if offered
    node_sci = evaluator.add_leaf(
        id="Science_Years_and_Lab",
        desc="States that 2 years of natural or physical science are required, including specification that one year must be a lab science if the high school offers it",
        parent=root,
        critical=True
    )
    sci_val = _val_or_missing(extr.science_years)
    lab_clause = ", and at least one must be a lab science if the high school offers it" if extr.mentions_lab_if_offered else ""
    claim_sci = f"{sci_val} years of natural or physical science are required{lab_clause}."
    src_sci = _merge_sources(extr.sources_science, extr.sources_general)
    addins_sci = (
        "Only mark as supported if the evidence explicitly indicates that 2 years of science are required "
        "AND that at least one year must be lab if the high school offers it. "
        "If the lab qualifier is missing from the claim, treat as not supported."
    )
    claims_and_sources.append((claim_sci, src_sci if src_sci else None, node_sci, addins_sci))

    # Social science years
    node_social = evaluator.add_leaf(
        id="Social_Science_Years",
        desc="States that 2 years of social science are required",
        parent=root,
        critical=True
    )
    social_val = _val_or_missing(extr.social_science_years)
    claim_social = f"{social_val} years of social science are required for NCAA Division I initial eligibility."
    src_social = _merge_sources(extr.sources_social, extr.sources_general)
    addins_social = (
        "Only support if the evidence explicitly indicates 2 years of social science are required."
    )
    claims_and_sources.append((claim_social, src_social if src_social else None, node_social, addins_social))

    # Extra English/Math/Science year (1)
    node_extra_ems = evaluator.add_leaf(
        id="Extra_English_Math_Science_Year",
        desc="States that 1 additional year of English, math, or natural/physical science is required",
        parent=root,
        critical=True
    )
    extra_ems_val = _val_or_missing(extr.extra_eng_math_sci_year)
    claim_extra_ems = f"An additional {extra_ems_val} year of English, math, or natural/physical science is required."
    src_extra_ems = _merge_sources(extr.sources_extra_ems, extr.sources_general)
    addins_extra_ems = (
        "Only support if the evidence explicitly indicates 1 additional year drawn from English, math, or natural/physical science."
    )
    claims_and_sources.append((claim_extra_ems, src_extra_ems if src_extra_ems else None, node_extra_ems, addins_extra_ems))

    # Additional four core years and allowed categories
    node_add4 = evaluator.add_leaf(
        id="Additional_Four_Core_Years",
        desc="States that 4 additional years of core courses from approved categories (which may include foreign language, philosophy, or non-doctrinal religion) are required",
        parent=root,
        critical=True
    )
    add4_val = _val_or_missing(extr.additional_four_core_years)
    cat_phrase = ", which may include foreign language, philosophy, or non-doctrinal religion" if extr.mentions_allowed_categories else ""
    claim_add4 = f"{add4_val} additional years of core courses from approved categories are required{cat_phrase}."
    src_add4 = _merge_sources(extr.sources_additional_four, extr.sources_general)
    addins_add4 = (
        "Only support if the evidence explicitly indicates 4 additional core years AND mentions that approved categories may include "
        "foreign language, philosophy, or non-doctrinal religion. If the categories qualifier is missing from the claim, do not support."
    )
    claims_and_sources.append((claim_add4, src_add4 if src_add4 else None, node_add4, addins_add4))

    # Eight-semester timeframe
    node_8sem = evaluator.add_leaf(
        id="Eight_Semester_Timeframe",
        desc="States that the 16 core courses must be completed within 8 semesters of high school",
        parent=root,
        critical=True
    )
    sem_val = _val_or_missing(extr.eight_semester_timeframe)
    claim_8sem = f"The 16 core courses must be completed within {sem_val} semesters of high school."
    src_8sem = _merge_sources(extr.sources_eight_sem, extr.sources_general)
    addins_8sem = (
        "Only support if the evidence explicitly indicates that all 16 core courses must be completed within 8 semesters."
    )
    claims_and_sources.append((claim_8sem, src_8sem if src_8sem else None, node_8sem, addins_8sem))

    # Minimum GPA for competition
    node_gpa = evaluator.add_leaf(
        id="Minimum_GPA_for_Competition",
        desc="States that a minimum 2.3 core-course GPA is required to compete in the first year (for students enrolling August 2016 or later)",
        parent=root,
        critical=True
    )
    gpa_val = _val_or_missing(extr.min_competition_gpa)
    claim_gpa = f"A minimum {gpa_val} core-course GPA is required to compete in the first year for students enrolling August 2016 or later."
    src_gpa = _merge_sources(extr.sources_gpa, extr.sources_general)
    addins_gpa = (
        "Only support if the evidence explicitly indicates that the minimum core-course GPA for competition is 2.3, "
        "and that this applies to students enrolling August 2016 or later."
    )
    claims_and_sources.append((claim_gpa, src_gpa if src_gpa else None, node_gpa, addins_gpa))

    # Sliding scale test scores
    node_sliding = evaluator.add_leaf(
        id="Sliding_Scale_Test_Scores",
        desc="Explains that SAT or ACT scores must meet a sliding scale based on core-course GPA",
        parent=root,
        critical=True
    )
    # Build claim text based on whether the answer explicitly mentioned the sliding scale
    claim_sliding = (
        "SAT or ACT scores must meet a sliding scale based on core-course GPA."
        if extr.sliding_scale_mentioned
        else "SAT or ACT test score requirements exist."
    )
    src_sliding = _merge_sources(extr.sources_tests, extr.sources_general)
    addins_sliding = (
        "Only support if the evidence explicitly shows a sliding scale (test score/GPA sliding index). "
        "If the claim text does not explicitly mention 'sliding scale', treat as not supported even if evidence shows it."
    )
    claims_and_sources.append((claim_sliding, src_sliding if src_sliding else None, node_sliding, addins_sliding))

    # Ten before seventh semester
    node_10_before7 = evaluator.add_leaf(
        id="Ten_Courses_Before_Seventh_Semester",
        desc="States that 10 of the 16 core courses must be completed before the start of the seventh semester (for students enrolling August 2016 or later)",
        parent=root,
        critical=True
    )
    ten_val = _val_or_missing(extr.ten_before_7th)
    claim_10_before7 = f"{ten_val} of the 16 core courses must be completed before the start of the seventh semester (for students enrolling August 2016 or later)."
    src_10_before7 = _merge_sources(extr.sources_ten_before7, extr.sources_general)
    addins_10_before7 = (
        "Only support if the evidence explicitly indicates that 10 of the 16 core courses must be completed "
        "before the start of the seventh semester (for students enrolling August 2016 or later). "
        "If the claim uses a different number or omits the 'before the start of the seventh semester' phrasing, do not support."
    )
    claims_and_sources.append((claim_10_before7, src_10_before7 if src_10_before7 else None, node_10_before7, addins_10_before7))

    # Seven of those 10 in English, math, science
    node_7_of_10 = evaluator.add_leaf(
        id="Seven_of_Ten_in_Core_Subjects",
        desc="States that at least 7 of those 10 early courses must be in English, math, and science",
        parent=root,
        critical=True
    )
    if extr.seven_of_ten_in_core_subjects:
        claim_7_of_10 = "At least 7 of those 10 early courses must be in English, mathematics, and natural/physical science."
    else:
        claim_7_of_10 = "A subset of the early core courses must be in English, math, and science."
    src_7_of_10 = _merge_sources(extr.sources_seven_of_ten, extr.sources_general)
    addins_7_of_10 = (
        "Only support if the evidence explicitly indicates that at least 7 of those 10 early courses must be in English, math, and science. "
        "If the claim does not explicitly state '7', do not support."
    )
    claims_and_sources.append((claim_7_of_10, src_7_of_10 if src_7_of_10 else None, node_7_of_10, addins_7_of_10))

    # Five-year eligibility clock (non-critical)
    node_clock = evaluator.add_leaf(
        id="Five_Year_Eligibility_Clock",
        desc="Explains that Division I student-athletes have five calendar years from first enrollment at any college to complete four seasons of competition",
        parent=root,
        critical=False
    )
    clock_val = _val_or_missing(extr.five_year_eligibility_clock)
    claim_clock = f"Division I student-athletes have {clock_val} calendar years from their first collegiate enrollment to complete four seasons of competition."
    src_clock = _merge_sources(extr.sources_five_year_clock, extr.sources_general)
    addins_clock = (
        "Support only if the evidence explicitly indicates the Division I five-year clock to use four seasons of competition starting from first full-time collegiate enrollment."
    )
    claims_and_sources.append((claim_clock, src_clock if src_clock else None, node_clock, addins_clock))

    # Transfer academic residence (non-critical)
    node_transfer = evaluator.add_leaf(
        id="Transfer_Academic_Residence",
        desc="Explains that transfers from one four-year institution to another typically must complete an academic year in residence before becoming eligible to compete",
        parent=root,
        critical=False
    )
    transfer_text = _val_or_missing(extr.transfer_academic_residence)
    claim_transfer = transfer_text if transfer_text != "[missing]" else "Transfers between four-year institutions typically must complete an academic year in residence before competing."
    src_transfer = _merge_sources(extr.sources_transfer, extr.sources_general)
    addins_transfer = (
        "Support only if the evidence explicitly indicates that, generally, four-year to four-year transfers must complete an academic year in residence before being eligible to compete (noting typical exceptions may exist)."
    )
    claims_and_sources.append((claim_transfer, src_transfer if src_transfer else None, node_transfer, addins_transfer))

    # Bachelor's degree credit hours (non-critical)
    node_120 = evaluator.add_leaf(
        id="Bachelor_Degree_Credit_Hours",
        desc="States that a bachelor's degree typically requires a minimum of 120 credit hours",
        parent=root,
        critical=False
    )
    credit_val = _val_or_missing(extr.bachelors_degree_credits)
    claim_120 = f"A bachelor's degree typically requires a minimum of {credit_val} credit hours."
    src_120 = _merge_sources(extr.sources_120_credits, extr.sources_general)
    addins_120 = (
        "Support only if the evidence indicates that a typical U.S. bachelor's degree minimum is 120 semester credit hours (or equivalent). "
        "If the claim uses a different number or is missing, do not support."
    )
    claims_and_sources.append((claim_120, src_120 if src_120 else None, node_120, addins_120))

    # Batch verify all leaves (parallel)
    await evaluator.batch_verify(claims_and_sources)


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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_ncaa_requirements(),
        template_class=NCAARequirementsExtraction,
        extraction_name="extracted_ncaa_requirements"
    )

    # Add ground truth context (for transparency; not used to auto-judge)
    evaluator.add_ground_truth({
        "expected_core_course_total": "16",
        "subject_breakdown": {
            "English": "4 years",
            "Math": "3 years (Algebra I level or higher)",
            "Science": "2 years (at least one lab if offered)",
            "Social Science": "2 years",
            "Additional English/Math/Science": "1 year",
            "Additional core from approved categories": "4 years (may include foreign language, philosophy, non-doctrinal religion)"
        },
        "timeline": {
            "eight_semesters": "All 16 core courses within 8 semesters",
            "early_completion": "10 of 16 before start of 7th semester; at least 7 of those 10 in English/Math/Science"
        },
        "gpa_and_tests": {
            "min_competition_gpa": "2.3 (for students enrolling Aug 2016 or later)",
            "tests": "Sliding scale based on core-course GPA"
        },
        "other_rules": {
            "five_year_clock": "5 calendar years to play 4 seasons from first collegiate enrollment",
            "transfer_residence": "Typically one academic year in residence for four-year to four-year transfers",
            "bachelor_credits": "Typically 120 credit hours"
        }
    }, gt_type="ground_truth")

    # Build verification leaves and verify
    await build_and_verify_ncaa_tree(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()