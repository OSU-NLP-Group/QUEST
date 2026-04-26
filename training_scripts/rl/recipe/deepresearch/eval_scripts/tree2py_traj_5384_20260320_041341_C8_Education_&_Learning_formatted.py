import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tx_uil6a_two_districts_2026_28"
TASK_DESCRIPTION = """
Identify two Texas public school districts that meet ALL of the following criteria:

Location and Classification:
- The district must be located in either Collin County, Denton County, or Travis County
- The district must be classified as UIL 6A for the 2026-28 realignment period
- The district must have high schools competing in UIL 6A District 5, 6, 7, or 24 according to the 2026-28 official district alignment

Enrollment and Size:
- The district must have total student enrollment between 40,000 and 50,000 students as of the 2023-2024 school year
- The district must operate at least 45 total campuses/schools
- The district must have at least 25 elementary schools
- The district must have at least 3 high schools

Academic Performance:
- The district must have a four-year graduation rate of at least 95% for the Class of 2023
- The district must have an average SAT score above 1050 for 2022-2023 graduates
- The district must have a dropout rate of 1% or lower for grades 9-12 during the 2022-2023 school year

Staffing and Programs:
- The district must have a student-to-teacher ratio of 15:1 or better (lower ratio)
- The district must have an average teacher experience of at least 10 years
- The district must offer Gifted and Talented programs with at least 15% of students participating in the program

For each district you identify, provide:
1. The complete official name of the district
2. The county in which it is located
3. The UIL 6A district number(s) for 2026-28 in which its schools compete
4. The total student enrollment (2023-2024)
5. The total number of campuses/schools
6. The four-year graduation rate (Class of 2023)
7. The average SAT score (2022-2023)
8. The dropout rate (2022-2023)
9. The student-to-teacher ratio
10. The average teacher experience in years
11. The Gifted and Talented program participation percentage
12. A reference URL from an official source (Texas Tribune Schools, Texas Education Agency, or the district's official website) supporting your answer
"""

UIL_ALLOWED_DISTRICTS = {"5", "6", "7", "24"}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DistrictMetrics(BaseModel):
    name: Optional[str] = None
    county: Optional[str] = None
    uil_classification: Optional[str] = None  # e.g., "UIL 6A"
    uil_district_numbers: List[str] = Field(default_factory=list)  # e.g., ["5", "24"]
    enrollment_2023_2024: Optional[str] = None
    total_campuses: Optional[str] = None
    elementary_schools: Optional[str] = None
    high_schools: Optional[str] = None
    graduation_rate_class_of_2023: Optional[str] = None
    avg_sat_2022_2023: Optional[str] = None
    dropout_rate_9_12_2022_2023: Optional[str] = None
    student_teacher_ratio: Optional[str] = None  # e.g., "14:1" or "14.3:1"
    avg_teacher_experience_years: Optional[str] = None
    gt_participation_percent: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class TwoDistrictsExtraction(BaseModel):
    districts: List[DistrictMetrics] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_two_districts() -> str:
    return """
    Extract up to the FIRST TWO Texas public school districts presented as meeting the criteria in the answer.
    For each district, return the following fields exactly as written in the answer (use strings for numbers and percentages):
    - name: Official district name (e.g., "Plano Independent School District")
    - county: The county named in the answer (e.g., "Collin County", "Denton County", or "Travis County")
    - uil_classification: The stated UIL classification for the 2026–28 period (e.g., "UIL 6A", "Class 6A")
    - uil_district_numbers: An array of the 2026–28 UIL 6A district number(s) (e.g., ["5"] or ["6","7"]). Normalize to digits only if a number is present (e.g., "District 06" -> "6"). If none stated, return [].
    - enrollment_2023_2024: Total student enrollment for 2023–2024 as written (e.g., "47,200")
    - total_campuses: Total number of campuses/schools as written (e.g., "52")
    - elementary_schools: Number of elementary schools as written (e.g., "28")
    - high_schools: Number of high schools as written (e.g., "3" or "4")
    - graduation_rate_class_of_2023: Four-year graduation rate for Class of 2023 as written (e.g., "96%")
    - avg_sat_2022_2023: Average SAT score for 2022–2023 graduates as written (e.g., "1085")
    - dropout_rate_9_12_2022_2023: Grades 9–12 dropout rate for 2022–2023 as written (e.g., "0.8%")
    - student_teacher_ratio: The student-to-teacher ratio as written (e.g., "14:1" or "14.5:1")
    - avg_teacher_experience_years: Average teacher experience in years as written (e.g., "10.4")
    - gt_participation_percent: Gifted & Talented participation percentage as written (e.g., "18%")
    - reference_urls: All URLs the answer cites for this district (Texas Tribune Schools, TEA, or district website). Include any general sources if they appear to support this district too. Extract actual URLs (resolve markdown links).

    Notes:
    - Do not invent values; if a field is missing in the answer, set it to null (or [] for the array).
    - Keep values exactly as written (strings), including commas and percent signs.
    - The 'districts' array should contain at most two entries (the first two districts described).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _str_or_placeholder(value: Optional[str], placeholder: str = "<missing>") -> str:
    return value if (value is not None and str(value).strip() != "") else placeholder


def _normalize_uil_numbers(nums: List[str]) -> List[str]:
    out: List[str] = []
    for n in nums or []:
        digits = "".join(ch for ch in str(n) if ch.isdigit())
        if digits:
            out.append(str(int(digits)))  # remove leading zeros
    # de-duplicate preserving order
    seen = set()
    ordered = []
    for x in out:
        if x not in seen:
            seen.add(x)
            ordered.append(x)
    return ordered


def _sources_or_none(d: DistrictMetrics) -> List[str]:
    return d.reference_urls or []


def _is_official_source(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
        if not host:
            return False
        if host.endswith("texastribune.org") and "schools" in path:
            return True
        if host.endswith("tea.texas.gov"):
            return True
        if host.endswith("txschools.gov"):
            return True
        # District official websites (heuristics):
        # Many Texas districts use domains containing "isd" or K-12 subdomains in .tx.us
        if "isd" in host:
            return True
        if host.endswith(".k12.tx.us") or host.endswith(".tx.us"):
            return True
        return False
    except Exception:
        return False


def _has_valid_official_url(urls: List[str]) -> bool:
    if not urls:
        return False
    return any(_is_official_source(u) for u in urls)


# --------------------------------------------------------------------------- #
# Verification per district                                                   #
# --------------------------------------------------------------------------- #
async def verify_one_district(
    evaluator: Evaluator,
    parent_node,
    district: DistrictMetrics,
    index_zero_based: int,
) -> None:
    idx = index_zero_based + 1
    dist_node = evaluator.add_parallel(
        id=f"district_{idx}",
        desc=("First qualifying district meeting all criteria" if idx == 1
              else "Second qualifying district meeting all criteria"),
        parent=parent_node,
        critical=False,
    )

    sources = _sources_or_none(district)
    uil_nums_norm = _normalize_uil_numbers(district.uil_district_numbers)
    uil_list_str = ", ".join(uil_nums_norm) if uil_nums_norm else "<none>"
    allowed_list_str = ", ".join(sorted(UIL_ALLOWED_DISTRICTS, key=lambda x: int(x)))

    # 1) Name (critical)
    name_leaf = evaluator.add_leaf(
        id=f"district_{idx}_name",
        desc="The district is correctly identified and named",
        parent=dist_node,
        critical=True,
    )
    name_val = _str_or_placeholder(district.name)
    await evaluator.verify(
        claim=f"The official name of the school district is '{name_val}'.",
        node=name_leaf,
        sources=sources,
        additional_instruction="Match the official district name on the cited page(s). Consider 'ISD' vs 'Independent School District' as equivalent if they clearly refer to the same entity.",
    )

    # 2) County (critical)
    county_leaf = evaluator.add_leaf(
        id=f"district_{idx}_county",
        desc="The district is located in Collin County, Denton County, or Travis County",
        parent=dist_node,
        critical=True,
    )
    county_val = _str_or_placeholder(district.county)
    await evaluator.verify(
        claim=f"This school district is located in {county_val} in Texas, which is one of Collin County, Denton County, or Travis County.",
        node=county_leaf,
        sources=sources,
        additional_instruction="If the district spans multiple counties, it still satisfies the requirement if one of the counties is Collin, Denton, or Travis.",
    )

    # 3) UIL classification 6A (critical)
    uil_class_leaf = evaluator.add_leaf(
        id=f"district_{idx}_uil_classification",
        desc="The district is classified as UIL 6A for 2026-28 realignment",
        parent=dist_node,
        critical=True,
    )
    uil_class_val = _str_or_placeholder(district.uil_classification)
    await evaluator.verify(
        claim=f"For the 2026–28 UIL realignment, this district's high school(s) compete in Class 6A (the answer states '{uil_class_val}').",
        node=uil_class_leaf,
        sources=sources,
        additional_instruction="Accept variations such as 'UIL 6A' or 'Conference 6A'. The evidence should clearly indicate 6A specifically for the 2026–28 alignment window.",
    )

    # 4) UIL 6A District membership within allowed set (critical)
    uil_d_leaf = evaluator.add_leaf(
        id=f"district_{idx}_uil_district",
        desc="The district has schools in UIL 6A District 5, 6, 7, or 24 (2026-28 alignment)",
        parent=dist_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"At least one 2026–28 UIL 6A high school from this district competes in District 5, 6, 7, or 24. "
            f"The answer lists district number(s): {uil_list_str}. Any overlap with {{{allowed_list_str}}} is sufficient."
        ),
        node=uil_d_leaf,
        sources=sources,
        additional_instruction="Look for UIL 2026–28 alignment pages or district/school athletics pages indicating district number(s). Passing requires at least one of 5, 6, 7, or 24.",
    )

    # 5) Enrollment in range 40k–50k (critical)
    enroll_leaf = evaluator.add_leaf(
        id=f"district_{idx}_enrollment_range",
        desc="The district has total enrollment between 40,000 and 50,000 students (2023-2024)",
        parent=dist_node,
        critical=True,
    )
    enroll_val = _str_or_placeholder(district.enrollment_2023_2024)
    await evaluator.verify(
        claim=(
            f"The district's total student enrollment for 2023–2024 is {enroll_val}, and this lies between 40,000 and 50,000."
        ),
        node=enroll_leaf,
        sources=sources,
        additional_instruction="Use the 2023–24 official enrollment as stated on TEA, Texas Tribune Schools, or the district website. Allow minor rounding differences.",
    )

    # 6) Total campuses ≥ 45 (critical)
    campuses_leaf = evaluator.add_leaf(
        id=f"district_{idx}_total_campuses",
        desc="The district operates at least 45 campuses/schools in total",
        parent=dist_node,
        critical=True,
    )
    campuses_val = _str_or_placeholder(district.total_campuses)
    await evaluator.verify(
        claim=f"The district operates {campuses_val} total campuses (schools), which is at least 45.",
        node=campuses_leaf,
        sources=sources,
        additional_instruction="Accept 'schools' and 'campuses' as equivalent. The evidence should clearly reflect the total count.",
    )

    # 7) Elementary ≥ 25 (critical)
    elem_leaf = evaluator.add_leaf(
        id=f"district_{idx}_elementary_count",
        desc="The district has at least 25 elementary schools",
        parent=dist_node,
        critical=True,
    )
    elem_val = _str_or_placeholder(district.elementary_schools)
    await evaluator.verify(
        claim=f"The district operates {elem_val} elementary schools, which is at least 25.",
        node=elem_leaf,
        sources=sources,
        additional_instruction="Elementary schools count may be labeled 'elementary campuses' on some pages.",
    )

    # 8) High schools ≥ 3 (critical)
    hs_leaf = evaluator.add_leaf(
        id=f"district_{idx}_high_school_count",
        desc="The district has at least 3 high schools",
        parent=dist_node,
        critical=True,
    )
    hs_val = _str_or_placeholder(district.high_schools)
    await evaluator.verify(
        claim=f"The district operates {hs_val} high schools, which is at least 3.",
        node=hs_leaf,
        sources=sources,
        additional_instruction="Count standalone high schools; specialty/early college campuses count only if designated as high schools by the official source.",
    )

    # 9) Graduation rate ≥ 95% (critical)
    grad_leaf = evaluator.add_leaf(
        id=f"district_{idx}_graduation_rate",
        desc="The district has a four-year graduation rate of at least 95% (Class of 2023)",
        parent=dist_node,
        critical=True,
    )
    grad_val = _str_or_placeholder(district.graduation_rate_class_of_2023)
    await evaluator.verify(
        claim=f"The four-year graduation rate for the Class of 2023 is {grad_val}, which is at least 95%.",
        node=grad_leaf,
        sources=sources,
        additional_instruction="Use Class of 2023 cohort rate from TEA/Texas Tribune/district. Allow rounding to nearest tenth or whole percent.",
    )

    # 10) SAT average > 1050 (critical)
    sat_leaf = evaluator.add_leaf(
        id=f"district_{idx}_sat_score",
        desc="The district has an average SAT score above 1050 (2022-2023 graduates)",
        parent=dist_node,
        critical=True,
    )
    sat_val = _str_or_placeholder(district.avg_sat_2022_2023)
    await evaluator.verify(
        claim=f"The average SAT score for 2022–2023 graduates is {sat_val}, which is above 1050.",
        node=sat_leaf,
        sources=sources,
        additional_instruction="Verify SAT average for the specified class year (2022–23). Accept composite average if clearly stated.",
    )

    # 11) Dropout rate ≤ 1% (critical)
    dropout_leaf = evaluator.add_leaf(
        id=f"district_{idx}_dropout_rate",
        desc="The district has a dropout rate of 1% or lower for grades 9-12 (2022-2023)",
        parent=dist_node,
        critical=True,
    )
    dropout_val = _str_or_placeholder(district.dropout_rate_9_12_2022_2023)
    await evaluator.verify(
        claim=f"The grades 9–12 dropout rate for 2022–2023 is {dropout_val}, which is 1% or lower.",
        node=dropout_leaf,
        sources=sources,
        additional_instruction="Use TEA-reported (or equivalent) 9–12 dropout rate for 2022–23. Accept ≤1.0% including values like 0.9%.",
    )

    # 12) Student-teacher ratio ≤ 15:1 (critical)
    ratio_leaf = evaluator.add_leaf(
        id=f"district_{idx}_student_teacher_ratio",
        desc="The district has a student-to-teacher ratio of 15:1 or better",
        parent=dist_node,
        critical=True,
    )
    ratio_val = _str_or_placeholder(district.student_teacher_ratio)
    await evaluator.verify(
        claim=f"The district's student-to-teacher ratio is {ratio_val}, which is 15:1 or better (i.e., no more than 15 students per teacher).",
        node=ratio_leaf,
        sources=sources,
        additional_instruction="If ratio is given as a number (e.g., 14.8), interpret as students per teacher. Treat 15.0 as meeting the requirement.",
    )

    # 13) Avg teacher experience ≥ 10 years (critical)
    exp_leaf = evaluator.add_leaf(
        id=f"district_{idx}_teacher_experience",
        desc="The district has average teacher experience of at least 10 years",
        parent=dist_node,
        critical=True,
    )
    exp_val = _str_or_placeholder(district.avg_teacher_experience_years)
    await evaluator.verify(
        claim=f"The district's average teacher experience is {exp_val} years, which is at least 10 years.",
        node=exp_leaf,
        sources=sources,
        additional_instruction="Use the district-wide average teacher experience in years. Allow rounding tolerance.",
    )

    # 14) Gifted/Talented participation ≥ 15% (critical)
    gt_leaf = evaluator.add_leaf(
        id=f"district_{idx}_gifted_talented",
        desc="The district offers Gifted and Talented programs with at least 15% student participation",
        parent=dist_node,
        critical=True,
    )
    gt_val = _str_or_placeholder(district.gt_participation_percent)
    await evaluator.verify(
        claim=f"The Gifted and Talented (GT) student participation rate is {gt_val}, which is at least 15%.",
        node=gt_leaf,
        sources=sources,
        additional_instruction="Use overall district GT participation percentage. Accept values like '15%' as meeting the threshold.",
    )

    # 15) Reference URL present and official (non-critical)
    ref_valid = _has_valid_official_url(sources)
    evaluator.add_custom_node(
        result=ref_valid,
        id=f"district_{idx}_reference_url",
        desc="A valid reference URL from Texas Tribune Schools, TEA, or district official website is provided",
        parent=dist_node,
        critical=False,  # per rubric: NON-CRITICAL
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
    # Initialize evaluator (root is parallel per rubric)
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

    # Extract districts from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_two_districts(),
        template_class=TwoDistrictsExtraction,
        extraction_name="two_districts_extraction",
    )

    # Ensure exactly two entries for downstream verification
    districts: List[DistrictMetrics] = list(extraction.districts or [])
    # Take first 2 if more
    if len(districts) > 2:
        districts = districts[:2]
    # Pad with empty placeholders if fewer than 2
    while len(districts) < 2:
        districts.append(DistrictMetrics())

    # Build subtree and verify each district
    await verify_one_district(evaluator, root, districts[0], 0)
    await verify_one_district(evaluator, root, districts[1], 1)

    # Return the evaluation summary
    return evaluator.get_summary()