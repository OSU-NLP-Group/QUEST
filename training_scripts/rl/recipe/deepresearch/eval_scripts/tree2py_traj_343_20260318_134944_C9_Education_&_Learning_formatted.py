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
TASK_ID = "us_school_districts_3_categories"
TASK_DESCRIPTION = """
Identify three distinct large public school districts in the United States for the 2024-2025 or 2025-2026 academic year, where each district represents a different category and meets all specified criteria for that category:

District Category 1: State's Largest District
Identify the largest public school district (by enrollment) in either Maryland, Texas, or Pennsylvania that meets ALL of the following criteria:
- Must be the single largest school district in its state by student enrollment
- Total student enrollment must exceed 150,000 students
- Must operate at least 200 schools or campuses
- Student demographic composition must show that at least three different racial/ethnic groups each represent 20% or more of the total student population
- Must have an academic calendar for 2024-2025 that includes at least 170 instructional days with a specified first day of school
- Must have publicly available enrollment and demographic data from the 2024-2025 school year
- Must have had a publicly reported enrollment figure or annual report released between July 2024 and March 2026

District Category 2: Exceptionally Diverse District
Identify a large public school district in Texas that meets ALL of the following criteria:
- Total student enrollment between 75,000 and 85,000 students
- Must operate at least 80 schools
- Student demographic composition must show that no single racial/ethnic group represents more than 30% of the total student population
- At least four different racial/ethnic groups must each represent at least 20% of the total student population
- Must be explicitly described in publicly available sources as one of the most culturally diverse districts in Texas or the United States
- Must have publicly available demographic data showing the racial/ethnic breakdown of students
- District must have been established before 1970

District Category 3: Medium-Large Eastern District
Identify a public school district in either New Jersey or Pennsylvania that meets ALL of the following criteria:
- Total student enrollment between 40,000 and 70,000 students
- Must operate at least 50 schools
- Must have a documented Title IX Coordinator with publicly available contact information
- Must have publicly available emergency school closing procedures and policies
- Must participate in the National School Lunch Program (NSLP)
- Must have publicly available information about special education services
- Must have had at least one news article or official announcement related to school closings, delays, or emergency procedures published between January 2024 and March 2026

For each identified district, provide:
1. The official name of the school district
2. The state in which it is located
3. A brief description of how it meets each required criterion for its category
4. At least one reference URL supporting each major criterion (enrollment size, demographic data, school count, programs, etc.)
"""

# Date windows
CAT1_REPORT_WINDOW = ("July 1, 2024", "March 31, 2026")
CAT3_NEWS_WINDOW = ("January 1, 2024", "March 31, 2026")

ALLOWED_STATES_CAT1 = {"Maryland", "Texas", "Pennsylvania", "MD", "TX", "PA"}
ALLOWED_STATES_CAT3 = {"New Jersey", "Pennsylvania", "NJ", "PA"}


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class GroupPercentage(BaseModel):
    group: Optional[str] = None
    percentage: Optional[str] = None  # Keep as string to maximize extraction compatibility


class CalendarInfo(BaseModel):
    instructional_days: Optional[str] = None
    first_day: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class RecentReport(BaseModel):
    date_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class TitleIXInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class DistrictCat1Extraction(BaseModel):
    district_name: Optional[str] = None
    state: Optional[str] = None

    enrollment_number: Optional[str] = None
    enrollment_urls: List[str] = Field(default_factory=list)

    largest_in_state_urls: List[str] = Field(default_factory=list)

    schools_count: Optional[str] = None
    schools_count_urls: List[str] = Field(default_factory=list)

    demographics: List[GroupPercentage] = Field(default_factory=list)
    demographics_urls: List[str] = Field(default_factory=list)

    calendar: CalendarInfo = Field(default_factory=CalendarInfo)

    data_urls_2024_25: List[str] = Field(default_factory=list)

    recent_report: RecentReport = Field(default_factory=RecentReport)


class DistrictCat2Extraction(BaseModel):
    district_name: Optional[str] = None
    state: Optional[str] = None

    enrollment_number: Optional[str] = None
    enrollment_urls: List[str] = Field(default_factory=list)

    schools_count: Optional[str] = None
    schools_count_urls: List[str] = Field(default_factory=list)

    demographics: List[GroupPercentage] = Field(default_factory=list)
    demographics_urls: List[str] = Field(default_factory=list)

    recognition_statement: Optional[str] = None
    recognition_urls: List[str] = Field(default_factory=list)

    established_year: Optional[str] = None
    established_urls: List[str] = Field(default_factory=list)


class DistrictCat3Extraction(BaseModel):
    district_name: Optional[str] = None
    state: Optional[str] = None

    enrollment_number: Optional[str] = None
    enrollment_urls: List[str] = Field(default_factory=list)

    schools_count: Optional[str] = None
    schools_count_urls: List[str] = Field(default_factory=list)

    title_ix: TitleIXInfo = Field(default_factory=TitleIXInfo)

    emergency_urls: List[str] = Field(default_factory=list)
    nslp_urls: List[str] = Field(default_factory=list)
    special_ed_urls: List[str] = Field(default_factory=list)

    recent_news: RecentReport = Field(default_factory=RecentReport)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_category_1() -> str:
    return """
    Extract the first (primary) district described for Category 1 (State's Largest District).
    Return a JSON object with the following fields:
    - district_name: official district name as written in the answer
    - state: the state name or abbreviation in which the district is located
    - enrollment_number: the total student enrollment number stated in the answer (string, keep formatting, e.g., "186,000" or "~186k")
    - enrollment_urls: array of URLs that support enrollment data (Google/official pages/annual reports, etc.)
    - largest_in_state_urls: array of URLs that support the claim that this district is the single largest in its state by student enrollment
    - schools_count: the number of schools or campuses operated (string, keep formatting)
    - schools_count_urls: array of URLs that support school/campus count
    - demographics: array of objects, each with:
        * group: racial/ethnic group name from the answer (e.g., Hispanic, Black, Asian, White, etc.)
        * percentage: the percentage string exactly as provided (e.g., "22%", "about 21.5%")
      Include as many as are provided in the answer.
    - demographics_urls: array of URLs that support demographic data for 2024-2025 (or most recent in answer if specified)
    - calendar: object with:
        * instructional_days: number of instructional days claimed for the 2024-2025 calendar (string)
        * first_day: the specific first day of school (string date as given)
        * urls: array of URLs supporting the 2024-2025 calendar info
    - data_urls_2024_25: array of URLs that directly provide publicly accessible enrollment and demographic data for the 2024-2025 school year
    - recent_report: object with:
        * date_text: the report/publication/announcement date as written in the answer (e.g., "Sept 2024", "February 12, 2025")
        * urls: array of URLs pointing to that report or announcement

    Rules:
    - Only extract URLs that appear in the answer text. Do not fabricate URLs.
    - If any item is missing in the answer, return null or an empty array accordingly.
    """


def prompt_extract_category_2() -> str:
    return """
    Extract the first (primary) district described for Category 2 (Exceptionally Diverse District in Texas).
    Return a JSON object with:
    - district_name
    - state
    - enrollment_number: total student enrollment number from the answer (string)
    - enrollment_urls: array of URLs supporting the enrollment number
    - schools_count: number of schools (string)
    - schools_count_urls: array of URLs supporting school count
    - demographics: array of { group, percentage } entries as written
    - demographics_urls: array of URLs supporting demographic breakdown
    - recognition_statement: the exact or summarized phrase that the district is "one of the most culturally diverse (in Texas or the US)"
    - recognition_urls: array of URLs supporting that recognition
    - established_year: the founding/establishment year as written (string)
    - established_urls: array of URLs supporting the establishment year

    Rules:
    - Use only URLs present in the answer.
    - If missing, set fields to null or empty arrays.
    """


def prompt_extract_category_3() -> str:
    return """
    Extract the first (primary) district described for Category 3 (Medium-Large Eastern District in NJ or PA).
    Return a JSON object with:
    - district_name
    - state
    - enrollment_number: total student enrollment number (string)
    - enrollment_urls: array of URLs supporting enrollment number
    - schools_count: number of schools (string)
    - schools_count_urls: array of URLs supporting school count
    - title_ix: object with:
        * name: Title IX Coordinator name (if provided in the answer)
        * email: Title IX Coordinator email (if provided)
        * phone: Title IX Coordinator phone (if provided)
        * urls: array of URLs that show the Title IX Coordinator info
    - emergency_urls: array of URLs showing publicly available emergency school closing procedures/policies
    - nslp_urls: array of URLs confirming NSLP participation
    - special_ed_urls: array of URLs with information about special education services
    - recent_news: object with:
        * date_text: the date from a news/official announcement about closings/delays/emergency procedures (as written)
        * urls: array of URLs pointing to the article/announcement

    Rules:
    - Only extract URLs that appear in the answer text.
    - If missing, set fields to null or empty arrays.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def parse_percent_to_float(p: Optional[str]) -> Optional[float]:
    if not p:
        return None
    # Extract first numeric token
    m = re.search(r"(\d{1,3}(?:\.\d+)?)", p.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_int_from_str(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"(\d{1,5})", s.replace(",", ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def urls_exist(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    return any(isinstance(u, str) and u.strip() for u in urls)


def union_urls(*url_lists: Optional[List[str]]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str):
                u2 = u.strip()
                if u2 and u2 not in seen:
                    seen.add(u2)
                    result.append(u2)
    return result


def top_groups_meeting_threshold(demos: List[GroupPercentage], threshold: float, k: int) -> List[GroupPercentage]:
    selected: List[GroupPercentage] = []
    for gp in demos:
        val = parse_percent_to_float(gp.percentage)
        if val is not None and val >= threshold:
            selected.append(gp)
        if len(selected) >= k:
            break
    return selected[:k]


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_category_1_nodes(evaluator: Evaluator, parent, c1: DistrictCat1Extraction) -> None:
    cat1_node = evaluator.add_parallel(
        id="district_category_1",
        desc="Identify and verify the State's Largest District meeting all specified criteria",
        parent=parent,
        critical=False
    )

    # State/location check (critical)
    all_cat1_sources = union_urls(
        c1.enrollment_urls,
        c1.largest_in_state_urls,
        c1.schools_count_urls,
        c1.demographics_urls,
        c1.calendar.urls if c1 and c1.calendar else [],
        c1.data_urls_2024_25,
        c1.recent_report.urls if c1 and c1.recent_report else [],
    )

    state_loc_leaf = evaluator.add_leaf(
        id="state_location_cat1",
        desc="District is located in Maryland, Texas, or Pennsylvania",
        parent=cat1_node,
        critical=True
    )
    loc_claim = f"The district '{c1.district_name or ''}' is located in the state '{c1.state or ''}', which is one of Maryland, Texas, or Pennsylvania."
    await evaluator.verify(
        claim=loc_claim,
        node=state_loc_leaf,
        sources=all_cat1_sources,
        additional_instruction="Verify the district's state. Consider MD=Maryland, TX=Texas, PA=Pennsylvania. Minor naming variants are acceptable."
    )

    # Largest in state (critical group)
    largest_node = evaluator.add_parallel(
        id="largest_in_state",
        desc="District is the single largest school district in its state by student enrollment",
        parent=cat1_node,
        critical=True
    )

    # URL presence as gating
    largest_url_presence = evaluator.add_custom_node(
        result=urls_exist(c1.largest_in_state_urls) or urls_exist(c1.enrollment_urls),
        id="enrollment_url_reference",
        desc="URL reference provided for enrollment ranking verification",
        parent=largest_node,
        critical=True
    )

    largest_verify_leaf = evaluator.add_leaf(
        id="enrollment_verification",
        desc="Evidence confirms this is the largest district in the state",
        parent=largest_node,
        critical=True
    )
    largest_sources = c1.largest_in_state_urls if urls_exist(c1.largest_in_state_urls) else c1.enrollment_urls
    largest_claim = f"According to the provided source(s), '{c1.district_name or ''}' is the single largest public school district in {c1.state or ''} by student enrollment."
    await evaluator.verify(
        claim=largest_claim,
        node=largest_verify_leaf,
        sources=largest_sources,
        additional_instruction="Confirm the page explicitly indicates this district is the largest in its state by enrollment (not just one of the largest)."
    )

    # Enrollment size > 150,000 (critical)
    enroll_node = evaluator.add_parallel(
        id="enrollment_size_cat1",
        desc="Total student enrollment exceeds 150,000 students",
        parent=cat1_node,
        critical=True
    )

    enroll_url_presence = evaluator.add_custom_node(
        result=urls_exist(c1.enrollment_urls),
        id="enrollment_url",
        desc="URL reference provided for enrollment data",
        parent=enroll_node,
        critical=True
    )

    enroll_num_leaf = evaluator.add_leaf(
        id="enrollment_number",
        desc="Specific enrollment number is provided and exceeds 150,000",
        parent=enroll_node,
        critical=True
    )
    enroll_claim = f"The total district enrollment is '{c1.enrollment_number or ''}', and this is greater than 150,000 students."
    await evaluator.verify(
        claim=enroll_claim,
        node=enroll_num_leaf,
        sources=c1.enrollment_urls,
        additional_instruction="Interpret numbers flexibly (ignore commas, accept approximations like ~ or about). Confirm the numeric value exceeds 150000."
    )

    # School count >= 200 (critical)
    schools_node = evaluator.add_parallel(
        id="school_count_cat1",
        desc="District operates at least 200 schools or campuses",
        parent=cat1_node,
        critical=True
    )

    schools_url_presence = evaluator.add_custom_node(
        result=urls_exist(c1.schools_count_urls),
        id="school_count_url",
        desc="URL reference provided for school count data",
        parent=schools_node,
        critical=True
    )

    schools_num_leaf = evaluator.add_leaf(
        id="school_number",
        desc="Specific number of schools is provided and is at least 200",
        parent=schools_node,
        critical=True
    )
    schools_claim = f"The district '{c1.district_name or ''}' operates '{c1.schools_count or ''}' schools or campuses, which is at least 200."
    await evaluator.verify(
        claim=schools_claim,
        node=schools_num_leaf,
        sources=c1.schools_count_urls,
        additional_instruction="Confirm the number of schools/campuses is >= 200. Accept reasonable synonyms like campuses or instructional sites."
    )

    # Demographic diversity: at least 3 groups >=20% (critical)
    demo_node = evaluator.add_parallel(
        id="demographic_diversity_cat1",
        desc="At least three different racial/ethnic groups each represent 20% or more of total enrollment",
        parent=cat1_node,
        critical=True
    )

    demo_url_presence = evaluator.add_custom_node(
        result=urls_exist(c1.demographics_urls),
        id="demographic_url",
        desc="URL reference provided for demographic data",
        parent=demo_node,
        critical=True
    )

    demo_breakdown_node = evaluator.add_parallel(
        id="demographic_breakdown",
        desc="Specific demographic percentages are provided showing at least three groups at 20%+",
        parent=demo_node,
        critical=True
    )

    groups20 = top_groups_meeting_threshold(c1.demographics or [], 20.0, 3)
    # Ensure 3 binary leaves
    for i in range(3):
        if i < len(groups20):
            gp = groups20[i]
            leaf_id = f"group_{i+1}_percentage"
            leaf = evaluator.add_leaf(
                id=leaf_id,
                desc=f"{['First','Second','Third'][i]} racial/ethnic group percentage is specified and is 20% or higher",
                parent=demo_breakdown_node,
                critical=True
            )
            claim = f"In '{c1.district_name or ''}', the group '{gp.group or ''}' represents '{gp.percentage or ''}' of students, which is at least 20%."
            await evaluator.verify(
                claim=claim,
                node=leaf,
                sources=c1.demographics_urls,
                additional_instruction="Verify the stated percentage on the page for this group is >= 20%. Minor rounding acceptable."
            )
        else:
            # Missing required group threshold -> fail this leaf explicitly
            evaluator.add_custom_node(
                result=False,
                id=f"group_{i+1}_percentage",
                desc=f"{['First','Second','Third'][i]} racial/ethnic group percentage is specified and is 20% or higher",
                parent=demo_breakdown_node,
                critical=True
            )

    # Academic calendar requirements (critical)
    cal_node = evaluator.add_parallel(
        id="academic_calendar_cat1",
        desc="Academic calendar for 2024-2025 includes at least 170 instructional days with specified first day",
        parent=cat1_node,
        critical=True
    )

    cal_url_presence = evaluator.add_custom_node(
        result=urls_exist(c1.calendar.urls if c1 and c1.calendar else []),
        id="calendar_url",
        desc="URL reference provided for academic calendar information",
        parent=cal_node,
        critical=True
    )

    cal_days_leaf = evaluator.add_leaf(
        id="instructional_days",
        desc="Number of instructional days is specified and is at least 170",
        parent=cal_node,
        critical=True
    )
    cal_days_claim = f"The 2024-2025 academic calendar for '{c1.district_name or ''}' specifies at least 170 instructional days (stated as '{(c1.calendar.instructional_days if c1 and c1.calendar else '')}')."
    await evaluator.verify(
        claim=cal_days_claim,
        node=cal_days_leaf,
        sources=c1.calendar.urls if c1 and c1.calendar else [],
        additional_instruction="Check the official 2024-2025 academic calendar for total instructional days; confirm it is >= 170."
    )

    cal_first_day_leaf = evaluator.add_leaf(
        id="first_day_specified",
        desc="First day of school for 2024-2025 is specified",
        parent=cal_node,
        critical=True
    )
    cal_first_claim = f"The 2024-2025 academic calendar specifies the first day of school as '{(c1.calendar.first_day if c1 and c1.calendar else '')}'."
    await evaluator.verify(
        claim=cal_first_claim,
        node=cal_first_day_leaf,
        sources=c1.calendar.urls if c1 and c1.calendar else [],
        additional_instruction="Verify the page explicitly lists a first day/start date for the 2024-2025 school year."
    )

    # Public data availability (critical)
    data_node = evaluator.add_parallel(
        id="public_data_availability_cat1",
        desc="Publicly available enrollment and demographic data from 2024-2025 school year",
        parent=cat1_node,
        critical=True
    )

    data_url_presence = evaluator.add_custom_node(
        result=urls_exist(c1.data_urls_2024_25),
        id="data_url",
        desc="URL reference provided for accessing the data",
        parent=data_node,
        critical=True
    )

    data_access_leaf = evaluator.add_leaf(
        id="data_accessibility",
        desc="Enrollment and demographic data for 2024-2025 are publicly accessible",
        parent=data_node,
        critical=True
    )
    data_claim = "The provided page(s) publicly present the district's enrollment and demographic data for the 2024-2025 school year (no login required)."
    await evaluator.verify(
        claim=data_claim,
        node=data_access_leaf,
        sources=c1.data_urls_2024_25,
        additional_instruction="Confirm that content is specifically for 2024-2025 and is visible without authentication."
    )

    # Recent report timing (critical)
    report_node = evaluator.add_parallel(
        id="recent_report_cat1",
        desc="Publicly reported enrollment figure or annual report released between July 2024 and March 2026",
        parent=cat1_node,
        critical=True
    )

    report_url_presence = evaluator.add_custom_node(
        result=urls_exist(c1.recent_report.urls if c1 and c1.recent_report else []),
        id="report_url",
        desc="URL reference provided for the report or announcement",
        parent=report_node,
        critical=True
    )

    report_date_leaf = evaluator.add_leaf(
        id="report_date",
        desc="Date of report or announcement is specified and falls within July 2024 to March 2026",
        parent=report_node,
        critical=True
    )
    report_claim = (
        f"The report/announcement (dated '{(c1.recent_report.date_text if c1 and c1.recent_report else '')}') "
        f"falls within the window from {CAT1_REPORT_WINDOW[0]} to {CAT1_REPORT_WINDOW[1]}."
    )
    await evaluator.verify(
        claim=report_claim,
        node=report_date_leaf,
        sources=c1.recent_report.urls if c1 and c1.recent_report else [],
        additional_instruction="Verify the page shows a clear publication/release/posted/updated date within July 1, 2024 and March 31, 2026."
    )


async def build_category_2_nodes(evaluator: Evaluator, parent, c2: DistrictCat2Extraction) -> None:
    cat2_node = evaluator.add_parallel(
        id="district_category_2",
        desc="Identify and verify the Exceptionally Diverse District meeting all specified criteria",
        parent=parent,
        critical=False
    )

    all_cat2_sources = union_urls(
        c2.enrollment_urls,
        c2.schools_count_urls,
        c2.demographics_urls,
        c2.recognition_urls,
        c2.established_urls,
    )

    # State (Texas) check (critical)
    state_leaf = evaluator.add_leaf(
        id="state_location_cat2",
        desc="District is located in Texas",
        parent=cat2_node,
        critical=True
    )
    state_claim = f"The district '{c2.district_name or ''}' is located in the state '{c2.state or ''}', which must be Texas."
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=all_cat2_sources,
        additional_instruction="Confirm the district is in Texas (TX)."
    )

    # Enrollment 75k–85k (critical)
    enroll_node = evaluator.add_parallel(
        id="enrollment_size_cat2",
        desc="Total student enrollment is between 75,000 and 85,000 students",
        parent=cat2_node,
        critical=True
    )

    enroll_url_presence = evaluator.add_custom_node(
        result=urls_exist(c2.enrollment_urls),
        id="enrollment_url_cat2",
        desc="URL reference provided for enrollment data",
        parent=enroll_node,
        critical=True
    )

    enroll_leaf = evaluator.add_leaf(
        id="enrollment_number_cat2",
        desc="Specific enrollment number is provided and falls within 75,000-85,000 range",
        parent=enroll_node,
        critical=True
    )
    enroll_claim = f"The district's total enrollment is '{c2.enrollment_number or ''}', which lies between 75,000 and 85,000 students."
    await evaluator.verify(
        claim=enroll_claim,
        node=enroll_leaf,
        sources=c2.enrollment_urls,
        additional_instruction="Normalize numbers (ignore commas/tilde). Confirm 75000 <= value <= 85000."
    )

    # School count >= 80 (critical)
    schools_node = evaluator.add_parallel(
        id="school_count_cat2",
        desc="District operates at least 80 schools",
        parent=cat2_node,
        critical=True
    )

    schools_url_presence = evaluator.add_custom_node(
        result=urls_exist(c2.schools_count_urls),
        id="school_count_url_cat2",
        desc="URL reference provided for school count data",
        parent=schools_node,
        critical=True
    )

    schools_leaf = evaluator.add_leaf(
        id="school_number_cat2",
        desc="Specific number of schools is provided and is at least 80",
        parent=schools_node,
        critical=True
    )
    schools_claim = f"The district '{c2.district_name or ''}' operates '{c2.schools_count or ''}' schools, which is at least 80."
    await evaluator.verify(
        claim=schools_claim,
        node=schools_leaf,
        sources=c2.schools_count_urls,
        additional_instruction="Confirm number of schools is >= 80."
    )

    # Diversity maximum: no group > 30% (critical)
    divmax_node = evaluator.add_parallel(
        id="diversity_maximum_cat2",
        desc="No single racial/ethnic group represents more than 30% of total enrollment",
        parent=cat2_node,
        critical=True
    )

    divmax_url_presence = evaluator.add_custom_node(
        result=urls_exist(c2.demographics_urls),
        id="diversity_url_cat2_max",
        desc="URL reference provided for demographic verification",
        parent=divmax_node,
        critical=True
    )

    divmax_leaf = evaluator.add_leaf(
        id="all_groups_under_30",
        desc="Demographic data confirms no group exceeds 30%",
        parent=divmax_node,
        critical=True
    )
    divmax_claim = f"For '{c2.district_name or ''}', the demographic breakdown shows that no single racial/ethnic group exceeds 30% of total enrollment."
    await evaluator.verify(
        claim=divmax_claim,
        node=divmax_leaf,
        sources=c2.demographics_urls,
        additional_instruction="Check all group percentages; ensure each group is <= 30%. Minor rounding acceptable."
    )

    # Diversity minimum: at least four groups >= 20% each (critical)
    divmin_node = evaluator.add_parallel(
        id="diversity_minimum_cat2",
        desc="At least four different racial/ethnic groups each represent at least 20% of total enrollment",
        parent=cat2_node,
        critical=True
    )

    divmin_url_presence = evaluator.add_custom_node(
        result=urls_exist(c2.demographics_urls),
        id="diversity_url_cat2_min",
        desc="URL reference provided for demographic data",
        parent=divmin_node,
        critical=True
    )

    four_breakdown = evaluator.add_parallel(
        id="four_groups_breakdown",
        desc="Specific demographic percentages show at least four groups at 20%+",
        parent=divmin_node,
        critical=True
    )

    groups20 = top_groups_meeting_threshold(c2.demographics or [], 20.0, 4)
    for i in range(4):
        if i < len(groups20):
            gp = groups20[i]
            leaf = evaluator.add_leaf(
                id=f"group_{i+1}_cat2",
                desc=f"{['First','Second','Third','Fourth'][i]} racial/ethnic group is specified at 20% or higher",
                parent=four_breakdown,
                critical=True
            )
            claim = f"In '{c2.district_name or ''}', the group '{gp.group or ''}' represents '{gp.percentage or ''}', which is at least 20%."
            await evaluator.verify(
                claim=claim,
                node=leaf,
                sources=c2.demographics_urls,
                additional_instruction="Verify this group's share is >= 20% on the cited page."
            )
        else:
            evaluator.add_custom_node(
                result=False,
                id=f"group_{i+1}_cat2",
                desc=f"{['First','Second','Third','Fourth'][i]} racial/ethnic group is specified at 20% or higher",
                parent=four_breakdown,
                critical=True
            )

    # Recognition as culturally diverse (critical)
    recog_node = evaluator.add_parallel(
        id="diversity_recognition_cat2",
        desc="Explicitly described in publicly available sources as one of the most culturally diverse districts",
        parent=cat2_node,
        critical=True
    )

    recog_url_presence = evaluator.add_custom_node(
        result=urls_exist(c2.recognition_urls),
        id="recognition_url",
        desc="URL reference provided for the diversity recognition statement",
        parent=recog_node,
        critical=True
    )

    recog_leaf = evaluator.add_leaf(
        id="recognition_statement",
        desc="Source explicitly states the district is among the most culturally diverse",
        parent=recog_node,
        critical=True
    )
    recog_claim = (
        f"The page explicitly describes '{c2.district_name or ''}' as one of the most culturally diverse districts "
        f"in Texas or the United States. Stated/summary in the answer: '{c2.recognition_statement or ''}'."
    )
    await evaluator.verify(
        claim=recog_claim,
        node=recog_leaf,
        sources=c2.recognition_urls,
        additional_instruction="Look for explicit phrasing like 'one of the most culturally diverse' or equivalent strength."
    )

    # Establishment before 1970 (critical)
    est_node = evaluator.add_parallel(
        id="establishment_date_cat2",
        desc="District was established before 1970",
        parent=cat2_node,
        critical=True
    )

    est_url_presence = evaluator.add_custom_node(
        result=urls_exist(c2.established_urls),
        id="founding_url",
        desc="URL reference provided for establishment date",
        parent=est_node,
        critical=True
    )

    est_leaf = evaluator.add_leaf(
        id="founding_year",
        desc="Specific establishment/founding year is provided and is before 1970",
        parent=est_node,
        critical=True
    )
    est_claim = f"The district was established/founded in '{c2.established_year or ''}', which is earlier than 1970."
    await evaluator.verify(
        claim=est_claim,
        node=est_leaf,
        sources=c2.established_urls,
        additional_instruction="Verify the founding/established year from the page and check that it is < 1970."
    )


async def build_category_3_nodes(evaluator: Evaluator, parent, c3: DistrictCat3Extraction) -> None:
    cat3_node = evaluator.add_parallel(
        id="district_category_3",
        desc="Identify and verify the Medium-Large Eastern District meeting all specified criteria",
        parent=parent,
        critical=False
    )

    all_cat3_sources = union_urls(
        c3.enrollment_urls,
        c3.schools_count_urls,
        c3.title_ix.urls if c3 and c3.title_ix else [],
        c3.emergency_urls,
        c3.nslp_urls,
        c3.special_ed_urls,
        c3.recent_news.urls if c3 and c3.recent_news else [],
    )

    # State (NJ or PA) check (critical)
    state_leaf = evaluator.add_leaf(
        id="state_location_cat3",
        desc="District is located in New Jersey or Pennsylvania",
        parent=cat3_node,
        critical=True
    )
    state_claim = f"The district '{c3.district_name or ''}' is located in the state '{c3.state or ''}', which must be New Jersey or Pennsylvania."
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=all_cat3_sources,
        additional_instruction="Confirm the state is NJ (New Jersey) or PA (Pennsylvania)."
    )

    # Enrollment 40k–70k (critical)
    enroll_node = evaluator.add_parallel(
        id="enrollment_size_cat3",
        desc="Total student enrollment is between 40,000 and 70,000 students",
        parent=cat3_node,
        critical=True
    )

    enroll_url_presence = evaluator.add_custom_node(
        result=urls_exist(c3.enrollment_urls),
        id="enrollment_url_cat3",
        desc="URL reference provided for enrollment data",
        parent=enroll_node,
        critical=True
    )

    enroll_leaf = evaluator.add_leaf(
        id="enrollment_number_cat3",
        desc="Specific enrollment number is provided and falls within 40,000-70,000 range",
        parent=enroll_node,
        critical=True
    )
    enroll_claim = f"The district's total enrollment is '{c3.enrollment_number or ''}', which lies between 40,000 and 70,000 students."
    await evaluator.verify(
        claim=enroll_claim,
        node=enroll_leaf,
        sources=c3.enrollment_urls,
        additional_instruction="Normalize numbers; confirm 40000 <= value <= 70000."
    )

    # School count >= 50 (critical)
    schools_node = evaluator.add_parallel(
        id="school_count_cat3",
        desc="District operates at least 50 schools",
        parent=cat3_node,
        critical=True
    )

    schools_url_presence = evaluator.add_custom_node(
        result=urls_exist(c3.schools_count_urls),
        id="school_count_url_cat3",
        desc="URL reference provided for school count data",
        parent=schools_node,
        critical=True
    )

    schools_leaf = evaluator.add_leaf(
        id="school_number_cat3",
        desc="Specific number of schools is provided and is at least 50",
        parent=schools_node,
        critical=True
    )
    schools_claim = f"The district '{c3.district_name or ''}' operates '{c3.schools_count or ''}' schools, which is at least 50."
    await evaluator.verify(
        claim=schools_claim,
        node=schools_leaf,
        sources=c3.schools_count_urls,
        additional_instruction="Confirm number of schools is >= 50."
    )

    # Title IX Coordinator (critical)
    tix_node = evaluator.add_parallel(
        id="title_ix_coordinator_cat3",
        desc="Documented Title IX Coordinator with publicly available contact information",
        parent=cat3_node,
        critical=True
    )

    tix_url_presence = evaluator.add_custom_node(
        result=urls_exist(c3.title_ix.urls if c3 and c3.title_ix else []),
        id="coordinator_url",
        desc="URL reference provided for Title IX Coordinator information",
        parent=tix_node,
        critical=True
    )

    tix_leaf = evaluator.add_leaf(
        id="coordinator_identified",
        desc="Title IX Coordinator name or contact information is provided",
        parent=tix_node,
        critical=True
    )
    tix_claim = (
        f"The provided page(s) list the district's Title IX Coordinator with at least one of: name ('{(c3.title_ix.name if c3 and c3.title_ix else '')}'), "
        f"email ('{(c3.title_ix.email if c3 and c3.title_ix else '')}'), or phone ('{(c3.title_ix.phone if c3 and c3.title_ix else '')}')."
    )
    await evaluator.verify(
        claim=tix_claim,
        node=tix_leaf,
        sources=c3.title_ix.urls if c3 and c3.title_ix else [],
        additional_instruction="Confirm the page explicitly provides Title IX Coordinator identification and contact info (any of name/email/phone)."
    )

    # Emergency procedures (critical)
    emer_node = evaluator.add_parallel(
        id="emergency_procedures_cat3",
        desc="Publicly available emergency school closing procedures and policies",
        parent=cat3_node,
        critical=True
    )

    emer_url_presence = evaluator.add_custom_node(
        result=urls_exist(c3.emergency_urls),
        id="procedures_url",
        desc="URL reference provided for emergency procedures",
        parent=emer_node,
        critical=True
    )

    emer_leaf = evaluator.add_leaf(
        id="procedures_documented",
        desc="Emergency closing procedures are documented and accessible",
        parent=emer_node,
        critical=True
    )
    emer_claim = "The provided page(s) document emergency school closing procedures and/or policies (e.g., weather closures, delays, notifications)."
    await evaluator.verify(
        claim=emer_claim,
        node=emer_leaf,
        sources=c3.emergency_urls,
        additional_instruction="Look for pages detailing procedures for closures/delays/emergencies (policy, how decisions/notifications occur)."
    )

    # NSLP participation (critical)
    nslp_node = evaluator.add_parallel(
        id="nslp_participation_cat3",
        desc="Participates in the National School Lunch Program (NSLP)",
        parent=cat3_node,
        critical=True
    )

    nslp_url_presence = evaluator.add_custom_node(
        result=urls_exist(c3.nslp_urls),
        id="nslp_url",
        desc="URL reference provided for NSLP participation",
        parent=nslp_node,
        critical=True
    )

    nslp_leaf = evaluator.add_leaf(
        id="nslp_confirmed",
        desc="Evidence confirms participation in NSLP",
        parent=nslp_node,
        critical=True
    )
    nslp_claim = "The district participates in the National School Lunch Program (NSLP)."
    await evaluator.verify(
        claim=nslp_claim,
        node=nslp_leaf,
        sources=c3.nslp_urls,
        additional_instruction="Confirm explicit mention of NSLP participation, eligibility, or references to USDA NSLP."
    )

    # Special education information (critical)
    sped_node = evaluator.add_parallel(
        id="special_education_cat3",
        desc="Publicly available information about special education services",
        parent=cat3_node,
        critical=True
    )

    sped_url_presence = evaluator.add_custom_node(
        result=urls_exist(c3.special_ed_urls),
        id="special_ed_url",
        desc="URL reference provided for special education information",
        parent=sped_node,
        critical=True
    )

    sped_leaf = evaluator.add_leaf(
        id="special_ed_documented",
        desc="Special education services information is publicly available",
        parent=sped_node,
        critical=True
    )
    sped_claim = "The provided page(s) publicly document the district's special education services (e.g., programs, supports, IEP services)."
    await evaluator.verify(
        claim=sped_claim,
        node=sped_leaf,
        sources=c3.special_ed_urls,
        additional_instruction="Confirm that the page describes services, departments, or resources for special education."
    )

    # Recent news window (critical)
    news_node = evaluator.add_parallel(
        id="recent_news_cat3",
        desc="At least one news article or official announcement about closings/delays from January 2024 to March 2026",
        parent=cat3_node,
        critical=True
    )

    news_url_presence = evaluator.add_custom_node(
        result=urls_exist(c3.recent_news.urls if c3 and c3.recent_news else []),
        id="news_url",
        desc="URL reference provided for the news article or announcement",
        parent=news_node,
        critical=True
    )

    news_date_leaf = evaluator.add_leaf(
        id="news_date",
        desc="Date of news article or announcement falls within January 2024 to March 2026",
        parent=news_node,
        critical=True
    )
    news_claim = (
        f"The article/announcement (dated '{(c3.recent_news.date_text if c3 and c3.recent_news else '')}') "
        f"concerns school closings/delays/emergency procedures and is dated between {CAT3_NEWS_WINDOW[0]} and {CAT3_NEWS_WINDOW[1]}."
    )
    await evaluator.verify(
        claim=news_claim,
        node=news_date_leaf,
        sources=c3.recent_news.urls if c3 and c3.recent_news else [],
        additional_instruction="Confirm the content concerns closures/delays/emergency procedures AND the date is within Jan 1, 2024 and Mar 31, 2026."
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
    Evaluate an answer for the three-district categories task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root: categories are independent
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

    # Extract structured info for each category (in parallel)
    c1_task = evaluator.extract(
        prompt=prompt_extract_category_1(),
        template_class=DistrictCat1Extraction,
        extraction_name="category_1_extraction"
    )
    c2_task = evaluator.extract(
        prompt=prompt_extract_category_2(),
        template_class=DistrictCat2Extraction,
        extraction_name="category_2_extraction"
    )
    c3_task = evaluator.extract(
        prompt=prompt_extract_category_3(),
        template_class=DistrictCat3Extraction,
        extraction_name="category_3_extraction"
    )

    c1, c2, c3 = await asyncio.gather(c1_task, c2_task, c3_task)

    # Build verification trees for each category
    await build_category_1_nodes(evaluator, root, c1)
    await build_category_2_nodes(evaluator, root, c2)
    await build_category_3_nodes(evaluator, root, c3)

    # Return the unified summary
    return evaluator.get_summary()