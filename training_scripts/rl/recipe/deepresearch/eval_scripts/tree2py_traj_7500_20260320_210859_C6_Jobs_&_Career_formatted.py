import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "superintendent_profile_verification"
TASK_DESCRIPTION = """
Identify the current superintendent of a school district who meets ALL of the following criteria:

District Requirements:
- The school district must have student enrollment exceeding 100,000
- The district must be located in Virginia, Maryland, or Florida

Appointment Timeline:
- The superintendent must have been officially appointed or sworn into their current position between January 1, 2023, and December 31, 2024 (inclusive)

Educational Credentials:
- Must hold an earned doctorate degree (Ed.D. or Ph.D.) in education or a related field
- The doctorate degree must have been completed between 2007 and 2010 (inclusive)
- The doctorate must have been earned from a public research university in the United States

Professional Experience:
- Must have at least 35 years of total experience in the education field
- Must have previously held administrative positions at the school or district level before becoming a superintendent

Provide the superintendent's name, the school district they lead, and detailed supporting information with URL references for each requirement.
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class SuperintendentExtraction(BaseModel):
    # Superintendent identity
    superintendent_name: Optional[str] = None
    superintendent_name_source_urls: List[str] = Field(default_factory=list)

    # District information
    district_name: Optional[str] = None
    district_state: Optional[str] = None
    district_verification_source_urls: List[str] = Field(default_factory=list)

    # Enrollment
    district_enrollment: Optional[str] = None
    district_enrollment_source_urls: List[str] = Field(default_factory=list)

    # Location
    district_location_source_urls: List[str] = Field(default_factory=list)

    # Superintendent-district connection
    connection_statement: Optional[str] = None
    connection_source_urls: List[str] = Field(default_factory=list)

    # Appointment timeline
    appointment_date: Optional[str] = None
    appointment_source_urls: List[str] = Field(default_factory=list)

    # Current position confirmation
    current_position_source_urls: List[str] = Field(default_factory=list)

    # Doctoral degree details
    degree_type: Optional[str] = None           # e.g., "Ed.D.", "Ph.D.", or text like "Doctor of Education"
    degree_field: Optional[str] = None          # e.g., "Educational Leadership"
    degree_university: Optional[str] = None
    degree_type_source_urls: List[str] = Field(default_factory=list)
    university_source_urls: List[str] = Field(default_factory=list)
    degree_completion_year: Optional[str] = None  # Accept single year or range like "2008" or "2007-2009"
    degree_completion_source_urls: List[str] = Field(default_factory=list)
    doctoral_education_source_urls: List[str] = Field(default_factory=list)

    # Professional experience
    experience_years: Optional[str] = None
    experience_source_urls: List[str] = Field(default_factory=list)

    # Administrative background
    prior_admin_positions: List[str] = Field(default_factory=list)
    prior_admin_source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_superintendent_profile() -> str:
    return """
    Extract the first superintendent profile presented in the answer that attempts to satisfy the task criteria.
    Return a single JSON object with the following fields (use null for missing scalars and [] for missing lists):

    Superintendent identity
    - superintendent_name: the full name of the superintendent (string)
    - superintendent_name_source_urls: URLs explicitly cited that confirm the superintendent's name and identity (array of URLs)

    District information
    - district_name: the official district name (string)
    - district_state: the U.S. state for the district (e.g., VA, Virginia, MD, Maryland, FL, Florida) (string)
    - district_verification_source_urls: URLs that confirm the district identity/basic info (array)

    Enrollment
    - district_enrollment: the stated total student enrollment number or textual expression (string)
    - district_enrollment_source_urls: URLs cited to support the enrollment (array)

    Location
    - district_location_source_urls: URLs cited to support the district's state/location (array)

    Superintendent-district connection
    - connection_statement: the statement that this person is superintendent of the identified district (string)
    - connection_source_urls: URLs confirming the superintendent leads the identified district (array)

    Appointment timeline
    - appointment_date: the appointment or swearing-in date (any text format; include at least the year) (string)
    - appointment_source_urls: URLs confirming the appointment/swearing-in date or timeframe (array)

    Current position
    - current_position_source_urls: URLs confirming the superintendent is currently in that role (array)

    Doctoral degree details
    - degree_type: the doctoral degree type (e.g., "Ed.D.", "Ph.D.", "Doctor of Education", "Doctor of Philosophy") (string)
    - degree_field: the field or program area (string)
    - degree_university: the degree-granting university (string)
    - degree_type_source_urls: URLs confirming the degree type (array)
    - university_source_urls: URLs about the degree university (array)
    - degree_completion_year: the completion year or year range (e.g., "2009" or "2007-2010") (string)
    - degree_completion_source_urls: URLs confirming completion year/timeframe (array)
    - doctoral_education_source_urls: any additional URLs documenting doctoral education background (array)

    Professional experience
    - experience_years: the total years of education experience as claimed (string)
    - experience_source_urls: URLs confirming years of experience (array)

    Administrative background
    - prior_admin_positions: list the prior administrative roles before becoming superintendent (array of role titles)
    - prior_admin_source_urls: URLs confirming administrative background (array)

    Rules:
    - Extract exactly what appears in the answer; do not invent.
    - If multiple candidates are listed, extract ONLY the first one in full.
    - For all URL fields, extract only explicit URLs present in the answer (plain links or within markdown).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
STATE_ALIASES = {
    "va": "VA",
    "virginia": "VA",
    "md": "MD",
    "maryland": "MD",
    "fl": "FL",
    "fla": "FL",
    "florida": "FL",
}


def normalize_state(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip().lower().replace(".", "")
    s = re.sub(r"\s+", " ", s)
    return STATE_ALIASES.get(s, raw.strip().upper())


def combine_sources(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if not isinstance(u, str):
                continue
            u = u.strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


def parse_numeric_quantity(text: Optional[str]) -> Optional[float]:
    """
    Try to interpret a textual quantity into a numeric value.
    Handles:
      - "150,000" -> 150000
      - "110k" or "110 k" -> 110000
      - "1.2 million" / "1.2m" -> 1_200_000
      - Plain 5+ digit sequences like "120000"
    Returns float or None.
    """
    if not text:
        return None
    t = text.lower().replace(",", " ").strip()
    # Patterns for 1.2 million, 110k, etc.
    m = re.search(r"(\d+(?:\.\d+)?)\s*(million|m)\b", t)
    if m:
        return float(m.group(1)) * 1_000_000
    m = re.search(r"(\d+(?:\.\d+)?)\s*(thousand|k)\b", t)
    if m:
        return float(m.group(1)) * 1_000
    # Comma/space separated big ints
    m = re.search(r"(\d[\d\s]{4,})", t)
    if m:
        digits = re.sub(r"\s+", "", m.group(1))
        try:
            return float(int(digits))
        except Exception:
            pass
    # Fallback: any integer
    m = re.search(r"(\d{2,})", t)
    if m:
        try:
            return float(int(m.group(1)))
        except Exception:
            pass
    return None


def enrollment_over_100k(enrollment_text: Optional[str]) -> bool:
    val = parse_numeric_quantity(enrollment_text)
    return bool(val is not None and val > 100_000)


def extract_years(text: Optional[str]) -> List[int]:
    if not text:
        return []
    years = re.findall(r"\b(19|20)\d{2}\b", text)
    # The regex above only returns the first two digits group; fix:
    years_full = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    out = []
    for y in years_full:
        try:
            yi = int(y)
            if 1900 <= yi <= 2100:
                out.append(yi)
        except Exception:
            pass
    # Also handle ranges like "2007-2010"
    for m in re.finditer(r"\b(19\d{2}|20\d{2})\s*[-/–]\s*(19\d{2}|20\d{2})\b", text):
        try:
            a = int(m.group(1))
            b = int(m.group(2))
            if 1900 <= a <= 2100 and 1900 <= b <= 2100:
                lo, hi = min(a, b), max(a, b)
                for yi in range(lo, hi + 1):
                    if yi not in out:
                        out.append(yi)
        except Exception:
            pass
    return sorted(set(out))


def any_year_in_range(text: Optional[str], start: int, end: int) -> bool:
    yrs = extract_years(text)
    return any(start <= y <= end for y in yrs)


def degree_year_in_2007_2010(text: Optional[str]) -> bool:
    return any_year_in_range(text, 2007, 2010)


def appointment_in_2023_2024(text: Optional[str]) -> bool:
    return any_year_in_range(text, 2023, 2024)


def experience_meets_35(text: Optional[str]) -> bool:
    if not text:
        return False
    # Try to extract the maximum numeric token
    nums = [int(n) for n in re.findall(r"\b(\d{1,3})\b", text)]
    if nums:
        if max(nums) >= 35:
            return True
    # Check textual hint "35+"
    if re.search(r"\b35\+|\bthirty[-\s]?five\b", text.lower()):
        return True
    return False


def degree_type_is_doctorate(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    patterns = [
        r"\bed\.?d\.?\b",
        r"\bph\.?d\.?\b",
        r"\bdoctor of education\b",
        r"\bdoctor of philosophy\b",
        r"\bdoctoral\b",
        r"\bdoctorate\b",
    ]
    return any(re.search(p, t) for p in patterns)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_district_identification(
    evaluator: Evaluator,
    parent,
    data: SuperintendentExtraction
) -> None:
    district_node = evaluator.add_parallel(
        id="District_Identification",
        desc="Identify and verify the school district meeting enrollment and location criteria",
        parent=parent,
        critical=False
    )

    # District_Name (critical)
    evaluator.add_custom_node(
        result=bool(data.district_name and data.district_name.strip()),
        id="District_Name",
        desc="School district name is provided",
        parent=district_node,
        critical=True
    )

    # District_Enrollment_Over_100000 (critical parallel)
    enroll_node = evaluator.add_parallel(
        id="District_Enrollment_Over_100000",
        desc="District enrollment exceeds 100,000 students with supporting enrollment data",
        parent=district_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(data.district_enrollment and str(data.district_enrollment).strip()),
        id="Enrollment_Number_Stated",
        desc="Specific enrollment number is stated",
        parent=enroll_node,
        critical=True
    )

    # Threshold met: simple logical check on extracted number
    thr_node = evaluator.add_leaf(
        id="Enrollment_Threshold_Met",
        desc="Stated enrollment is greater than 100,000",
        parent=enroll_node,
        critical=True
    )
    thr_claim = f"The stated enrollment value '{data.district_enrollment}' is greater than 100,000."
    await evaluator.verify(
        claim=thr_claim,
        node=thr_node,
        additional_instruction=(
            "Only check the numeric meaning of the provided text; "
            "if the text implies a number over 100,000 (e.g., 110k, 1.2 million), consider it greater than 100,000."
        )
    )

    # Enrollment source URL(s) verification: confirm 'exceeds 100,000' from sources
    enroll_src_node = evaluator.add_leaf(
        id="Enrollment_Source_URL",
        desc="URL reference provided for enrollment data",
        parent=enroll_node,
        critical=True
    )
    enroll_sources = data.district_enrollment_source_urls
    enroll_claim = f"The student enrollment for {data.district_name or 'the district'} exceeds 100,000."
    await evaluator.verify(
        claim=enroll_claim,
        node=enroll_src_node,
        sources=enroll_sources,
        additional_instruction=(
            "Verify from the page content that the district's total enrollment is above 100,000. "
            "Allow minor differences in exact counts but it must clearly be > 100,000."
        ),
    )

    # District_Location_In_Specified_States (critical parallel)
    loc_node = evaluator.add_parallel(
        id="District_Location_In_Specified_States",
        desc="District is located in Virginia, Maryland, or Florida",
        parent=district_node,
        critical=True
    )

    normalized_state = normalize_state(data.district_state)

    evaluator.add_custom_node(
        result=bool(data.district_state and data.district_state.strip()),
        id="State_Identified",
        desc="State is clearly identified",
        parent=loc_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=normalized_state in {"VA", "MD", "FL"} if normalized_state else False,
        id="State_In_Required_List",
        desc="Identified state is VA, MD, or FL",
        parent=loc_node,
        critical=True
    )

    loc_src_node = evaluator.add_leaf(
        id="Location_Source_URL",
        desc="URL reference provided for district location",
        parent=loc_node,
        critical=True
    )
    location_sources = combine_sources(data.district_location_source_urls, data.district_verification_source_urls)
    loc_claim = f"The school district {data.district_name or ''} is located in the U.S. state of {normalized_state or data.district_state or 'the specified state'}."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_src_node,
        sources=location_sources,
        additional_instruction="Confirm from the source that the district is in the stated U.S. state (Virginia/VA, Maryland/MD, or Florida/FL)."
    )

    # District_Verification_Source (critical)
    dv_sources = combine_sources(
        data.district_verification_source_urls,
        data.district_location_source_urls,
        data.district_enrollment_source_urls
    )
    if dv_sources:
        dv_node = evaluator.add_leaf(
            id="District_Verification_Source",
            desc="URL reference confirming district identity and basic information",
            parent=district_node,
            critical=True
        )
        dv_claim = f"This source page is about the school district named '{data.district_name}'."
        await evaluator.verify(
            claim=dv_claim,
            node=dv_node,
            sources=dv_sources,
            additional_instruction="The page should clearly be about the identified school district (official site or authoritative source)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="District_Verification_Source",
            desc="URL reference confirming district identity and basic information",
            parent=district_node,
            critical=True
        )


async def build_superintendent_appointment(
    evaluator: Evaluator,
    parent,
    data: SuperintendentExtraction
) -> None:
    appoint_node = evaluator.add_parallel(
        id="Superintendent_Appointment_Verification",
        desc="Verify the current superintendent and appointment timeline",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(data.superintendent_name and data.superintendent_name.strip()),
        id="Superintendent_Name",
        desc="Full name of the superintendent is provided",
        parent=appoint_node,
        critical=True
    )

    # District_Superintendent_Connection (critical parallel)
    conn_node = evaluator.add_parallel(
        id="District_Superintendent_Connection",
        desc="Connection between named superintendent and identified district is established",
        parent=appoint_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(data.connection_statement and data.connection_statement.strip()),
        id="Connection_Stated",
        desc="Relationship between superintendent and district is clearly stated",
        parent=conn_node,
        critical=True
    )

    conn_src_node = evaluator.add_leaf(
        id="Connection_Source_URL",
        desc="URL reference confirming superintendent leads this district",
        parent=conn_node,
        critical=True
    )
    conn_sources = combine_sources(data.connection_source_urls, data.current_position_source_urls)
    conn_claim = f"{data.superintendent_name or 'The named person'} is the superintendent of {data.district_name or 'the district'}."
    await evaluator.verify(
        claim=conn_claim,
        node=conn_src_node,
        sources=conn_sources,
        additional_instruction="Confirm the page shows this person serves as superintendent of the identified district."
    )

    # Appointment timeline (critical parallel)
    ap_node = evaluator.add_parallel(
        id="Appointment_Date_2023_2024",
        desc="Superintendent was appointed or sworn in between January 1, 2023 and December 31, 2024",
        parent=appoint_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(data.appointment_date and data.appointment_date.strip()),
        id="Appointment_Date_Stated",
        desc="Specific appointment or swearing-in date is provided",
        parent=ap_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=appointment_in_2023_2024(data.appointment_date),
        id="Date_Within_Range",
        desc="Stated date falls between January 1, 2023 and December 31, 2024",
        parent=ap_node,
        critical=True
    )

    ap_src_node = evaluator.add_leaf(
        id="Appointment_Source_URL",
        desc="URL reference for appointment date",
        parent=ap_node,
        critical=True
    )
    ap_claim = (
        f"{data.superintendent_name or 'The superintendent'} was appointed or sworn in as superintendent of "
        f"{data.district_name or 'the district'} within the years 2023 or 2024."
    )
    await evaluator.verify(
        claim=ap_claim,
        node=ap_src_node,
        sources=data.appointment_source_urls,
        additional_instruction="The page should show the appointment/swearing-in occurred in 2023 or 2024."
    )

    # Current position confirmation (critical)
    cur_pos_node = evaluator.add_leaf(
        id="Current_Position_Source",
        desc="URL reference confirming current superintendent status",
        parent=appoint_node,
        critical=True
    )
    cur_pos_claim = f"{data.superintendent_name or 'The named person'} is currently serving as superintendent of {data.district_name or 'the district'}."
    await evaluator.verify(
        claim=cur_pos_claim,
        node=cur_pos_node,
        sources=data.current_position_source_urls,
        additional_instruction="Confirm the person is the current superintendent (present-tense or current roster/leadership page)."
    )


async def build_education_and_experience(
    evaluator: Evaluator,
    parent,
    data: SuperintendentExtraction
) -> None:
    edu_prof_node = evaluator.add_parallel(
        id="Educational_And_Professional_Credentials",
        desc="Verify doctorate degree and professional experience requirements",
        parent=parent,
        critical=False
    )

    # Doctorate_Degree_Verification (critical parallel)
    doc_node = evaluator.add_parallel(
        id="Doctorate_Degree_Verification",
        desc="Verification that superintendent holds required doctorate degree",
        parent=edu_prof_node,
        critical=True
    )

    # Degree type Ed.D or Ph.D (critical parallel)
    deg_type_node = evaluator.add_parallel(
        id="Degree_Type_EdD_or_PhD",
        desc="Degree type is Ed.D. or Ph.D. in education or related field",
        parent=doc_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(data.degree_type and degree_type_is_doctorate(data.degree_type)),
        id="Degree_Specification",
        desc="Specific degree type (Ed.D. or Ph.D.) is stated",
        parent=deg_type_node,
        critical=True
    )

    deg_type_src_leaf = evaluator.add_leaf(
        id="Degree_Type_Source_URL",
        desc="URL reference for degree type",
        parent=deg_type_node,
        critical=True
    )
    deg_type_claim = (
        f"{data.superintendent_name or 'The superintendent'} holds a doctoral degree "
        f"({data.degree_type or 'doctoral'}) in {data.degree_field or 'a related field'} "
        f"from {data.degree_university or 'the named university'}."
    )
    await evaluator.verify(
        claim=deg_type_claim,
        node=deg_type_src_leaf,
        sources=data.degree_type_source_urls,
        additional_instruction="Verify that the page confirms the person holds a doctoral degree and its type (Ed.D./Ph.D.)."
    )

    # University public research institution (critical parallel)
    uni_node = evaluator.add_parallel(
        id="University_Public_Research_Institution",
        desc="Doctorate was earned from a public research university in the United States",
        parent=doc_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(data.degree_university and data.degree_university.strip()),
        id="University_Name_Provided",
        desc="Name of degree-granting university is stated",
        parent=uni_node,
        critical=True
    )

    pub_res_leaf = evaluator.add_leaf(
        id="Public_Research_University_Confirmed",
        desc="University is confirmed as a public research institution",
        parent=uni_node,
        critical=True
    )
    pub_res_claim = (
        f"{data.degree_university or 'The university'} is a public research university located in the United States."
    )
    await evaluator.verify(
        claim=pub_res_claim,
        node=pub_res_leaf,
        sources=data.university_source_urls,
        additional_instruction=(
            "The page(s) should indicate the university is public (not private) and engaged in research "
            "(e.g., Carnegie R1/R2 or otherwise clearly described as a public research university in the U.S.)."
        )
    )

    evaluator.add_custom_node(
        result=bool(data.university_source_urls and len(data.university_source_urls) > 0),
        id="University_Source_URL",
        desc="URL reference for university information",
        parent=uni_node,
        critical=True
    )

    # Degree completion 2007-2010 (critical parallel)
    deg_comp_node = evaluator.add_parallel(
        id="Degree_Completion_2007_2010",
        desc="Doctorate degree was completed between 2007 and 2010 (inclusive)",
        parent=doc_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(data.degree_completion_year and data.degree_completion_year.strip()),
        id="Completion_Years_Stated",
        desc="Completion year or year range is provided",
        parent=deg_comp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=degree_year_in_2007_2010(data.degree_completion_year),
        id="Years_Within_2007_2010",
        desc="Stated completion timeframe falls within 2007-2010",
        parent=deg_comp_node,
        critical=True
    )

    comp_src_leaf = evaluator.add_leaf(
        id="Completion_Source_URL",
        desc="URL reference for degree completion timeframe",
        parent=deg_comp_node,
        critical=True
    )
    comp_claim = (
        f"{data.superintendent_name or 'The superintendent'} completed their doctoral degree "
        f"({data.degree_type or 'doctoral'}) at {data.degree_university or 'the university'} "
        f"in the timeframe 2007–2010."
    )
    await evaluator.verify(
        claim=comp_claim,
        node=comp_src_leaf,
        sources=data.degree_completion_source_urls,
        additional_instruction="Confirm from the source that the doctoral completion year is 2007, 2008, 2009, or 2010."
    )

    # Doctoral_Education_Source (critical leaf)
    doc_ed_sources = combine_sources(
        data.doctoral_education_source_urls,
        data.degree_type_source_urls,
        data.university_source_urls,
        data.degree_completion_source_urls
    )
    if doc_ed_sources:
        doc_ed_leaf = evaluator.add_leaf(
            id="Doctoral_Education_Source",
            desc="URL reference documenting doctoral education background",
            parent=doc_node,
            critical=True
        )
        doc_ed_claim = (
            f"{data.superintendent_name or 'The superintendent'} holds a doctoral degree from "
            f"{data.degree_university or 'the university'}."
        )
        await evaluator.verify(
            claim=doc_ed_claim,
            node=doc_ed_leaf,
            sources=doc_ed_sources,
            additional_instruction="The page should document the doctoral education background of the person."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Doctoral_Education_Source",
            desc="URL reference documenting doctoral education background",
            parent=doc_node,
            critical=True
        )

    # Professional_Experience_35_Plus_Years (critical parallel)
    exp_node = evaluator.add_parallel(
        id="Professional_Experience_35_Plus_Years",
        desc="Superintendent has at least 35 years of experience in education field",
        parent=edu_prof_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(data.experience_years and data.experience_years.strip()),
        id="Years_Of_Experience_Stated",
        desc="Total years of education experience is stated",
        parent=exp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=experience_meets_35(data.experience_years),
        id="Experience_Meets_35_Year_Minimum",
        desc="Stated experience is 35 years or more",
        parent=exp_node,
        critical=True
    )

    exp_src_leaf = evaluator.add_leaf(
        id="Experience_Duration_Source_URL",
        desc="URL reference for years of experience",
        parent=exp_node,
        critical=True
    )
    exp_claim = (
        f"{data.superintendent_name or 'The superintendent'} has at least 35 years of experience in education."
    )
    await evaluator.verify(
        claim=exp_claim,
        node=exp_src_leaf,
        sources=data.experience_source_urls,
        additional_instruction="Confirm from the source that total education experience is 35+ years."
    )

    # Administrative_Background_School_District_Level (critical parallel)
    admin_node = evaluator.add_parallel(
        id="Administrative_Background_School_District_Level",
        desc="Superintendent held prior administrative positions at school or district level",
        parent=edu_prof_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(data.prior_admin_positions and len(data.prior_admin_positions) > 0),
        id="Administrative_Positions_Identified",
        desc="Prior administrative roles are identified",
        parent=admin_node,
        critical=True
    )

    admin_level_leaf = evaluator.add_leaf(
        id="School_District_Level_Confirmed",
        desc="Identified roles were at school or district administrative level",
        parent=admin_node,
        critical=True
    )
    positions_str = ", ".join(data.prior_admin_positions) if data.prior_admin_positions else "administrative roles"
    admin_level_claim = (
        f"Before becoming superintendent, {data.superintendent_name or 'the superintendent'} held "
        f"administrative positions at the school or district level, such as: {positions_str}."
    )
    await evaluator.verify(
        claim=admin_level_claim,
        node=admin_level_leaf,
        sources=data.prior_admin_source_urls,
        additional_instruction="Verify that the roles are administrative and at the school or district level (e.g., principal, assistant principal, director, area superintendent, etc.)."
    )

    evaluator.add_custom_node(
        result=bool(data.prior_admin_source_urls and len(data.prior_admin_source_urls) > 0),
        id="Administrative_Background_Source_URL",
        desc="URL reference for administrative career background",
        parent=admin_node,
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the superintendent profile verification task.
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

    # Extraction
    extracted: SuperintendentExtraction = await evaluator.extract(
        prompt=prompt_extract_superintendent_profile(),
        template_class=SuperintendentExtraction,
        extraction_name="superintendent_profile_extraction",
    )

    # Optional: record normalized/derived helpers
    derived_info = {
        "normalized_state": normalize_state(extracted.district_state),
        "enrollment_parsed_value": parse_numeric_quantity(extracted.district_enrollment),
        "appointment_years": extract_years(extracted.appointment_date),
        "degree_years": extract_years(extracted.degree_completion_year),
    }
    evaluator.add_custom_info(derived_info, info_type="derived_fields", info_name="derived_fields")

    # Build tree under a main wrapper node to mirror rubric
    main_node = evaluator.add_parallel(
        id="Superintendent_Profile_Verification",
        desc="Complete verification of a superintendent's identity and credentials based on specified criteria",
        parent=root,
        critical=False
    )

    # Subtrees
    await build_district_identification(evaluator, main_node, extracted)
    await build_superintendent_appointment(evaluator, main_node, extracted)
    await build_education_and_experience(evaluator, main_node, extracted)

    # Return summary
    return evaluator.get_summary()