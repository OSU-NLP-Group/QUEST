import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "rrisd_comprehensive_2026_03"
TASK_DESCRIPTION = (
    "Research Round Rock Independent School District (Round Rock ISD) in Texas and provide comprehensive information "
    "about the district and its current superintendent as of March 2026. Specifically, provide: "
    "(1) The district's total student enrollment for the 2024-25 school year; "
    "(2) The total number of schools operated by the district; "
    "(3) The full name of the current superintendent; "
    "(4) The type and field of the superintendent's doctoral degree; "
    "(5) The name of the university where the superintendent earned their doctoral degree; "
    "(6) Evidence of the superintendent's prior experience as a K-12 classroom teacher; "
    "(7) Evidence of the superintendent's prior administrative experience; "
    "(8) The year the superintendent began serving in their current role at Round Rock ISD; "
    "(9) The superintendent's current annual base salary; "
    "(10) Confirmation that Round Rock ISD is an independent school district serving K-12 students; "
    "(11) The district's most recent Texas Education Agency accountability rating; "
    "(12) The name of the university where the superintendent earned their master's degree; "
    "(13) The field or major of the superintendent's bachelor's degree; "
    "(14) At least one reference URL from the official Round Rock ISD website that supports the provided information. "
    "For each piece of information, include the reference URL(s) where it was found."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RRISDExtraction(BaseModel):
    # Currency / As-of
    as_of_statement: Optional[str] = None
    as_of_urls: List[str] = Field(default_factory=list)

    # District information
    enrollment_2024_25: Optional[str] = None
    enrollment_urls: List[str] = Field(default_factory=list)

    num_schools: Optional[str] = None
    num_schools_urls: List[str] = Field(default_factory=list)

    district_type_k12: Optional[str] = None
    district_type_urls: List[str] = Field(default_factory=list)

    tea_rating: Optional[str] = None
    tea_rating_year: Optional[str] = None
    tea_rating_urls: List[str] = Field(default_factory=list)

    chief_positions: List[str] = Field(default_factory=list)
    chief_positions_urls: List[str] = Field(default_factory=list)

    # Superintendent identity & education
    superintendent_name: Optional[str] = None
    superintendent_urls: List[str] = Field(default_factory=list)

    doctorate_type_field: Optional[str] = None
    doctorate_university: Optional[str] = None
    doctorate_year: Optional[str] = None
    doctorate_urls: List[str] = Field(default_factory=list)

    masters_university: Optional[str] = None
    masters_field: Optional[str] = None
    masters_year: Optional[str] = None
    masters_urls: List[str] = Field(default_factory=list)

    bachelors_field: Optional[str] = None
    bachelors_university: Optional[str] = None
    bachelors_urls: List[str] = Field(default_factory=list)

    # Experience & tenure
    teaching_experience_desc: Optional[str] = None
    teaching_experience_urls: List[str] = Field(default_factory=list)

    admin_experience_desc: Optional[str] = None
    admin_experience_urls: List[str] = Field(default_factory=list)

    career_start_year: Optional[str] = None
    career_start_urls: List[str] = Field(default_factory=list)

    start_year_rrisd: Optional[str] = None
    start_year_rrisd_urls: List[str] = Field(default_factory=list)

    # Compensation
    base_salary: Optional[str] = None
    base_salary_urls: List[str] = Field(default_factory=list)

    # Sourcing
    official_rrisd_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_rrisd() -> str:
    return """
Extract the specific, structured information below from the answer (do NOT invent). For each requested datum, also extract all supporting URLs that the answer explicitly cites.

Return a single JSON object with these fields (use null for any missing value and [] for missing lists):

Currency ("as of" context):
- as_of_statement: The statement in the answer that indicates information is current as of March 2026 (or notes "most recent available data" where March 2026 data aren't available), quoted or paraphrased exactly from the answer.
- as_of_urls: URLs supporting time-currency or recency if provided in the answer.

District information:
- enrollment_2024_25: The total student enrollment for the 2024-25 school year exactly as stated.
- enrollment_urls: URLs supporting this enrollment figure.
- num_schools: The total number of schools the district operates exactly as stated.
- num_schools_urls: URLs supporting the number of schools.
- district_type_k12: A short confirmation phrase that Round Rock ISD is an independent school district serving K-12 students (e.g., "Independent school district serving K-12").
- district_type_urls: URLs supporting the district type/K-12 confirmation.
- tea_rating: The most recent TEA accountability rating letter (e.g., "B").
- tea_rating_year: The rating year (e.g., "2024-25" or "2024").
- tea_rating_urls: URLs supporting the rating.
- chief_positions: A list of titles of district chief-level executive roles mentioned (e.g., "Chief of Schools", "Chief Academic Officer").
- chief_positions_urls: URLs where these roles are listed/described.

Superintendent identity & education:
- superintendent_name: Full name of the current superintendent.
- superintendent_urls: URLs supporting the current superintendent identity.
- doctorate_type_field: Type and field of the superintendent’s doctoral degree (e.g., "Ed.D. in Educational Leadership").
- doctorate_university: University that conferred the doctoral degree (as written).
- doctorate_year: The year the doctorate was earned (if present), else null.
- doctorate_urls: URLs supporting the doctoral details.
- masters_university: University that conferred the master’s degree (as written).
- masters_field: Field/major of the master’s degree if stated; else null.
- masters_year: The year the master’s was earned (if present), else null.
- masters_urls: URLs supporting the master’s details.
- bachelors_field: Field/major of the bachelor’s degree (e.g., "Physics and Chemistry").
- bachelors_university: University granting the bachelor’s degree if stated; else null.
- bachelors_urls: URLs supporting the bachelor’s details.

Experience & tenure:
- teaching_experience_desc: Text summarizing prior K-12 classroom teaching experience, preferably indicating middle school science if present.
- teaching_experience_urls: URLs supporting teaching experience.
- admin_experience_desc: Text summarizing prior administrative experience (e.g., principal, assistant superintendent).
- admin_experience_urls: URLs supporting administrative experience.
- career_start_year: The year the superintendent began their education career (e.g., "2002") if the answer states it, else null.
- career_start_urls: URLs supporting the career start year.
- start_year_rrisd: The year the superintendent began serving as Round Rock ISD superintendent (e.g., "2021") if the answer states it, else null.
- start_year_rrisd_urls: URLs supporting the superintendent start year at RRISD.

Compensation:
- base_salary: The superintendent’s current annual base salary exactly as stated (include currency symbol if present).
- base_salary_urls: URLs supporting the salary figure.

Sourcing:
- official_rrisd_urls: From all extracted URLs above, include those that come from the official Round Rock ISD domain (roundrockisd.org) or its subdomains.

Rules:
- Extract EXACTLY what appears in the answer for each value; if a value is not present, set it to null; if URLs are not provided, set the corresponding URL list to [].
- Do not guess or add information not present in the answer.
- Preserve number formatting and degree titles as presented in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def has_value_and_sources(value: Optional[str], urls: List[str]) -> bool:
    return bool(value and str(value).strip()) and bool(urls and len(urls) > 0)


def gather_all_urls(extract: RRISDExtraction) -> List[str]:
    urls: List[str] = []
    data = extract.dict()
    for k, v in data.items():
        if k.endswith("_urls") and isinstance(v, list):
            urls.extend([u for u in v if isinstance(u, str) and u.strip()])
    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def is_official_rrisd_url(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.endswith("roundrockisd.org") or netloc.endswith(".roundrockisd.org") or netloc == "roundrockisd.org"
    except Exception:
        return False


def any_official_rrisd_url(extract: RRISDExtraction) -> bool:
    all_urls = gather_all_urls(extract)
    return any(is_official_rrisd_url(u) for u in all_urls) or any(is_official_rrisd_url(u) for u in extract.official_rrisd_urls)


def try_parse_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"(-?\d[0-9,]*)", s.replace("\u2013", "-").replace("\u2014", "-"))
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except Exception:
        return None


def try_parse_money_to_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    # Capture digits possibly after a currency symbol
    m = re.search(r"(\d[\d,]{2,})", s)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except Exception:
        return None


def text_includes_edd_educational_leadership(s: Optional[str]) -> bool:
    if not s:
        return False
    t = s.lower().strip()
    has_edd = ("ed.d" in t) or ("doctor of education" in t) or ("edd" in t)
    has_el = ("educational leadership" in t) or ("education leadership" in t)
    return has_edd and has_el


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_asof_node(evaluator: Evaluator, parent, ex: RRISDExtraction) -> None:
    # Leaf: As-of date / currency stated in the answer
    asof_leaf = evaluator.add_leaf(
        id="asof_march_2026_declared",
        desc="Response explicitly indicates information is current as of March 2026 (or clearly states most recent available data dates).",
        parent=parent,
        critical=True,
    )
    claim = (
        "The answer explicitly indicates that the information is current as of March 2026, "
        "or clearly states the most recent available data dates where March 2026 values are not available."
    )
    await evaluator.verify(
        claim=claim,
        node=asof_leaf,
        additional_instruction="Look only at the provided answer text to determine if it states 'as of March 2026' or an equivalent explicit time-currency note."
    )


async def build_district_info_nodes(evaluator: Evaluator, parent, ex: RRISDExtraction) -> None:
    district_node = evaluator.add_parallel(
        id="district_information",
        desc="Provide district-level facts and ensure they meet the stated constraints.",
        parent=parent,
        critical=True,
    )

    # Enrollment 2024-25: approx 46,000–47,000
    enr_node = evaluator.add_parallel(
        id="enrollment_2024_25_checks",
        desc="District Enrollment (2024-25) within Constraint Range",
        parent=district_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_value_and_sources(ex.enrollment_2024_25, ex.enrollment_urls),
        id="enrollment_value_present",
        desc="Enrollment value for 2024-25 is provided with at least one source URL",
        parent=enr_node,
        critical=True,
    )
    enr_range_leaf = evaluator.add_leaf(
        id="enrollment_in_range_46k_47k",
        desc="2024-25 enrollment is approximately 46,000–47,000",
        parent=enr_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The district's total student enrollment for the 2024-25 school year is between 46,000 and 47,000 students (inclusive).",
        node=enr_range_leaf,
        sources=ex.enrollment_urls,
        additional_instruction="Check numbers shown on the cited page(s); approximate rounding is acceptable if the stated figure clearly falls within 46,000–47,000."
    )
    enr_exact_leaf = evaluator.add_leaf(
        id="enrollment_value_supported",
        desc="The specific 2024-25 enrollment value stated in the answer is supported by the sources",
        parent=enr_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The 2024-25 total student enrollment is {ex.enrollment_2024_25}.",
        node=enr_exact_leaf,
        sources=ex.enrollment_urls,
        additional_instruction="Confirm the exact enrollment figure appears on the provided URLs."
    )

    # Total number of schools equals 58
    sch_node = evaluator.add_parallel(
        id="num_schools_checks",
        desc="Total Number of Schools equals Constraint Value",
        parent=district_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_value_and_sources(ex.num_schools, ex.num_schools_urls),
        id="num_schools_present",
        desc="Total number of schools is provided with at least one source URL",
        parent=sch_node,
        critical=True,
    )
    schools_leaf = evaluator.add_leaf(
        id="num_schools_equals_58",
        desc="The district operates 58 schools",
        parent=sch_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The district operates 58 schools.",
        node=schools_leaf,
        sources=ex.num_schools_urls,
        additional_instruction="Verify the page(s) explicitly show that Round Rock ISD has 58 schools."
    )

    # District type: ISD serving K-12
    type_node = evaluator.add_parallel(
        id="district_type_checks",
        desc="District Type (ISD serving K-12)",
        parent=district_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_value_and_sources(ex.district_type_k12, ex.district_type_urls),
        id="district_type_present",
        desc="District type/K-12 confirmation provided with at least one source URL",
        parent=type_node,
        critical=True,
    )
    type_leaf = evaluator.add_leaf(
        id="isd_serves_k12",
        desc="Round Rock ISD is an independent school district serving K-12 students",
        parent=type_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Round Rock ISD is an independent school district serving K-12 students.",
        node=type_leaf,
        sources=ex.district_type_urls,
        additional_instruction="Look for explicit wording or clear implication that RRISD is an ISD and serves K-12."
    )

    # TEA accountability rating: 'B' for 2024-25
    tea_node = evaluator.add_parallel(
        id="tea_rating_checks",
        desc="Most Recent TEA Accountability Rating matches Constraint",
        parent=district_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_value_and_sources(ex.tea_rating, ex.tea_rating_urls),
        id="tea_rating_present",
        desc="TEA accountability rating is provided with at least one source URL",
        parent=tea_node,
        critical=True,
    )
    tea_leaf = evaluator.add_leaf(
        id="tea_rating_b_2024_25",
        desc="District has a 'B' rating for 2024-25",
        parent=tea_node,
        critical=True,
    )
    claim_tea = "Round Rock ISD received a 'B' rating in the most recent TEA accountability ratings for the 2024-25 cycle (or latest corresponding year used)."
    await evaluator.verify(
        claim=claim_tea,
        node=tea_leaf,
        sources=ex.tea_rating_urls,
        additional_instruction="Confirm the rating letter 'B' on the page(s); tolerate minor year labeling differences if clearly equivalent to 2024-25."
    )

    # Chief-level executive positions: multiple exist
    chief_node = evaluator.add_parallel(
        id="chief_positions_checks",
        desc="Chief-level Executive Leadership Positions Exist",
        parent=district_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(len(ex.chief_positions) >= 2 and len(ex.chief_positions_urls) > 0),
        id="chief_positions_present_multiple",
        desc="At least two chief-level executive leadership titles are listed with source URLs",
        parent=chief_node,
        critical=True,
    )
    chief_leaf = evaluator.add_leaf(
        id="chief_positions_supported",
        desc="District lists multiple chief-level executive leadership roles",
        parent=chief_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The cited page(s) show multiple chief-level executive leadership positions in Round Rock ISD (e.g., several roles titled 'Chief ...').",
        node=chief_leaf,
        sources=ex.chief_positions_urls,
        additional_instruction="You should see at least two different 'Chief' roles or similar executive titles on the page(s)."
    )


async def build_superintendent_identity_education_nodes(evaluator: Evaluator, parent, ex: RRISDExtraction) -> None:
    sup_node = evaluator.add_parallel(
        id="superintendent_identity_education",
        desc="Superintendent Identity & Education",
        parent=parent,
        critical=True,
    )

    # Current superintendent name
    name_node = evaluator.add_parallel(
        id="superintendent_name_checks",
        desc="Current Superintendent Name",
        parent=sup_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_value_and_sources(ex.superintendent_name, ex.superintendent_urls),
        id="superintendent_name_present",
        desc="Superintendent name provided with at least one source URL",
        parent=name_node,
        critical=True,
    )
    name_leaf = evaluator.add_leaf(
        id="superintendent_name_supported",
        desc="Sources support the stated current superintendent name",
        parent=name_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The current superintendent of Round Rock ISD is {ex.superintendent_name}.",
        node=name_leaf,
        sources=ex.superintendent_urls,
        additional_instruction="Confirm that the page(s) identify this person as the current superintendent."
    )

    # Doctoral degree: type and field (Ed.D. in Educational Leadership)
    doc_type_node = evaluator.add_parallel(
        id="doctorate_type_field_checks",
        desc="Doctoral Degree Type and Field matches Constraint",
        parent=sup_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=text_includes_edd_educational_leadership(ex.doctorate_type_field),
        id="doctorate_type_field_matches_constraint",
        desc="Doctoral degree is Ed.D. in Educational Leadership per the answer text",
        parent=doc_type_node,
        critical=True,
    )
    doc_type_leaf = evaluator.add_leaf(
        id="doctorate_type_field_supported",
        desc="Sources support that the superintendent holds an Ed.D. in Educational Leadership",
        parent=doc_type_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The superintendent holds an Ed.D. (Doctor of Education) in Educational Leadership.",
        node=doc_type_leaf,
        sources=ex.doctorate_urls,
        additional_instruction="Look for degree type (Ed.D.) and field (Educational Leadership) explicitly on the cited page(s)."
    )

    # Doctoral degree university (and Texas university constraint)
    doc_u_node = evaluator.add_parallel(
        id="doctorate_university_checks",
        desc="Doctoral Degree University (and Texas-university constraint)",
        parent=sup_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_value_and_sources(ex.doctorate_university, ex.doctorate_urls),
        id="doctorate_university_present",
        desc="Doctoral degree university provided with at least one source URL",
        parent=doc_u_node,
        critical=True,
    )
    doc_u_leaf = evaluator.add_leaf(
        id="doctorate_university_supported",
        desc="Sources support the stated doctoral university",
        parent=doc_u_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The superintendent earned the doctoral degree from {ex.doctorate_university}.",
        node=doc_u_leaf,
        sources=ex.doctorate_urls,
        additional_instruction="Confirm the page(s) explicitly mention this doctoral institution for the superintendent."
    )
    doc_tx_leaf = evaluator.add_leaf(
        id="doctorate_university_in_texas",
        desc="Doctoral institution is a Texas university",
        parent=doc_u_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The doctoral institution is a Texas university (i.e., located in Texas, USA).",
        node=doc_tx_leaf,
        sources=ex.doctorate_urls,
        additional_instruction="Look for explicit city/state or wording indicating the university is in Texas; if the university name includes 'Texas', that is sufficient."
    )

    # Master's degree university (and Texas university constraint)
    mas_u_node = evaluator.add_parallel(
        id="masters_university_checks",
        desc="Master's Degree University (and Texas-university constraint)",
        parent=sup_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_value_and_sources(ex.masters_university, ex.masters_urls),
        id="masters_university_present",
        desc="Master's degree university provided with at least one source URL",
        parent=mas_u_node,
        critical=True,
    )
    mas_u_leaf = evaluator.add_leaf(
        id="masters_university_supported",
        desc="Sources support the stated master's university",
        parent=mas_u_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The superintendent earned the master's degree from {ex.masters_university}.",
        node=mas_u_leaf,
        sources=ex.masters_urls,
        additional_instruction="Confirm the page(s) explicitly mention this master's institution for the superintendent."
    )
    mas_tx_leaf = evaluator.add_leaf(
        id="masters_university_in_texas",
        desc="Master's institution is a Texas university",
        parent=mas_u_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The master's degree institution is a Texas university (i.e., located in Texas, USA).",
        node=mas_tx_leaf,
        sources=ex.masters_urls,
        additional_instruction="Look for explicit city/state or wording indicating the university is in Texas; if the university name includes 'Texas', that is sufficient."
    )

    # Master's earned before doctorate (order constraint)
    order_leaf = evaluator.add_leaf(
        id="masters_before_doctorate_order_supported",
        desc="Sources support that the master's was earned before the doctoral degree",
        parent=sup_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The superintendent earned the master's degree before earning the doctoral degree.",
        node=order_leaf,
        sources=(ex.masters_urls + ex.doctorate_urls),
        additional_instruction="If years are listed, ensure the master's year is earlier. If only narrative sequence is provided, it must clearly indicate master's precedes doctorate."
    )

    # Bachelor's degree field/major matches physics and chemistry (STEM)
    bach_node = evaluator.add_parallel(
        id="bachelors_field_checks",
        desc="Bachelor's Degree Field/Major matches Constraint",
        parent=sup_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_value_and_sources(ex.bachelors_field, ex.bachelors_urls),
        id="bachelors_field_present",
        desc="Bachelor's degree field/major provided with at least one source URL",
        parent=bach_node,
        critical=True,
    )
    bach_leaf = evaluator.add_leaf(
        id="bachelors_field_physics_chemistry_supported",
        desc="Sources support that the bachelor's field includes physics and chemistry",
        parent=bach_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The superintendent's bachelor's degree is in a STEM field that includes physics and chemistry (e.g., physics and chemistry, or a double major/minor combination including both).",
        node=bach_leaf,
        sources=ex.bachelors_urls,
        additional_instruction="Look for explicit mention of both 'physics' and 'chemistry' in the bachelor's field or combination."
    )


async def build_superintendent_experience_tenure_nodes(evaluator: Evaluator, parent, ex: RRISDExtraction) -> None:
    exp_node = evaluator.add_parallel(
        id="superintendent_experience_tenure",
        desc="Superintendent Experience & Tenure",
        parent=parent,
        critical=True,
    )

    # K-12 teaching experience (middle school science)
    teach_node = evaluator.add_parallel(
        id="teaching_experience_checks",
        desc="Evidence of K-12 Classroom Teaching Experience (middle school science)",
        parent=exp_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_value_and_sources(ex.teaching_experience_desc, ex.teaching_experience_urls),
        id="teaching_experience_present",
        desc="Teaching experience description provided with at least one source URL",
        parent=teach_node,
        critical=True,
    )
    teach_leaf = evaluator.add_leaf(
        id="teaching_experience_supported",
        desc="Sources support prior work as a K-12 classroom teacher (specifically middle school science)",
        parent=teach_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The superintendent previously worked as a K-12 classroom teacher, specifically as a middle school science teacher.",
        node=teach_leaf,
        sources=ex.teaching_experience_urls,
        additional_instruction="Look for explicit wording indicating 'middle school science' teaching experience."
    )

    # Administrative experience (principal / assistant superintendent, etc.)
    admin_node = evaluator.add_parallel(
        id="administrative_experience_checks",
        desc="Evidence of Administrative Experience (principal / assistant superintendent, etc.)",
        parent=exp_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_value_and_sources(ex.admin_experience_desc, ex.admin_experience_urls),
        id="administrative_experience_present",
        desc="Administrative experience description provided with at least one source URL",
        parent=admin_node,
        critical=True,
    )
    admin_leaf = evaluator.add_leaf(
        id="administrative_experience_supported",
        desc="Sources support prior administrative roles (e.g., principal and/or assistant superintendent)",
        parent=admin_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The superintendent held administrative positions such as principal and/or assistant superintendent prior to the current role.",
        node=admin_leaf,
        sources=ex.admin_experience_urls,
        additional_instruction="Look for explicit job titles such as 'principal', 'assistant superintendent', or equivalent district-level administrative roles."
    )

    # Career start year satisfies 20+ years constraint (began in 2002)
    career_node = evaluator.add_parallel(
        id="career_start_checks",
        desc="Career Start Year satisfies 20+ years constraint",
        parent=exp_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_value_and_sources(ex.career_start_year, ex.career_start_urls),
        id="career_start_present",
        desc="Career start year provided with at least one source URL",
        parent=career_node,
        critical=True,
    )
    career_2002_leaf = evaluator.add_leaf(
        id="career_started_in_2002_supported",
        desc="Sources support that the superintendent began the education career in 2002",
        parent=career_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The superintendent began their education career in 2002.",
        node=career_2002_leaf,
        sources=ex.career_start_urls,
        additional_instruction="Look for an explicit year '2002' indicating the start of the education career."
    )
    # Derived 20+ years by March 2026 (non-URL custom check based on extracted year)
    start_year_parsed = try_parse_int(ex.career_start_year)
    evaluator.add_custom_node(
        result=(start_year_parsed is not None and (2026 - start_year_parsed) >= 20),
        id="career_experience_20plus_by_2026",
        desc="Derived check: 20+ years of experience as of March 2026 based on the stated career start year",
        parent=career_node,
        critical=True,
    )

    # Start year in current RRISD superintendent role satisfies 3+ years (since 2021)
    start_rrisd_node = evaluator.add_parallel(
        id="start_year_rrisd_checks",
        desc="Start Year in Current Superintendent Role satisfies 3+ years constraint",
        parent=exp_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_value_and_sources(ex.start_year_rrisd, ex.start_year_rrisd_urls),
        id="start_year_rrisd_present",
        desc="Start year as RRISD superintendent provided with at least one source URL",
        parent=start_rrisd_node,
        critical=True,
    )
    start_rrisd_leaf = evaluator.add_leaf(
        id="start_year_rrisd_since_2021_supported",
        desc="Sources support that the superintendent has served since 2021 (or earlier), implying 3+ years by March 2026",
        parent=start_rrisd_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The superintendent began serving as Round Rock ISD superintendent in 2021.",
        node=start_rrisd_leaf,
        sources=ex.start_year_rrisd_urls,
        additional_instruction="Look for a specific start year of 2021; this inherently implies 3+ years by March 2026."
    )


async def build_compensation_node(evaluator: Evaluator, parent, ex: RRISDExtraction) -> None:
    comp_node = evaluator.add_parallel(
        id="compensation",
        desc="Provide current base salary information meeting stated constraints.",
        parent=parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_value_and_sources(ex.base_salary, ex.base_salary_urls),
        id="base_salary_present",
        desc="Current annual base salary provided with at least one source URL",
        parent=comp_node,
        critical=True,
    )
    # Range check: $300,000–$400,000
    salary_int = try_parse_money_to_int(ex.base_salary)
    evaluator.add_custom_node(
        result=(salary_int is not None and 300_000 <= salary_int <= 400_000),
        id="base_salary_in_range_300k_400k",
        desc="Current annual base salary is between $300,000 and $400,000",
        parent=comp_node,
        critical=True,
    )
    salary_leaf = evaluator.add_leaf(
        id="base_salary_value_supported",
        desc="Sources support the stated current annual base salary",
        parent=comp_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The superintendent's current annual base salary is {ex.base_salary}.",
        node=salary_leaf,
        sources=ex.base_salary_urls,
        additional_instruction="Confirm the exact base salary value appears on the provided URL(s). Dollar formatting variations are acceptable if the numeric amount matches."
    )


async def build_sourcing_references_node(evaluator: Evaluator, parent, ex: RRISDExtraction) -> None:
    src_node = evaluator.add_parallel(
        id="sourcing_references",
        desc="Ensure each required datum is supported by URL citations, including at least one official RRISD URL.",
        parent=parent,
        critical=True,
    )

    # Check per-field sources provided (cover key fields)
    required_fields_with_urls: List[Tuple[Optional[str], List[str]]] = [
        (ex.enrollment_2024_25, ex.enrollment_urls),
        (ex.num_schools, ex.num_schools_urls),
        (ex.district_type_k12, ex.district_type_urls),
        (ex.tea_rating, ex.tea_rating_urls),
        (ex.superintendent_name, ex.superintendent_urls),
        (ex.doctorate_type_field, ex.doctorate_urls),
        (ex.doctorate_university, ex.doctorate_urls),
        (ex.masters_university, ex.masters_urls),
        (ex.bachelors_field, ex.bachelors_urls),
        (ex.teaching_experience_desc, ex.teaching_experience_urls),
        (ex.admin_experience_desc, ex.admin_experience_urls),
        (ex.start_year_rrisd, ex.start_year_rrisd_urls),
        (ex.base_salary, ex.base_salary_urls),
    ]
    all_have_sources = all(has_value_and_sources(val, urls) for val, urls in required_fields_with_urls)

    evaluator.add_custom_node(
        result=all_have_sources,
        id="per_field_source_urls_provided",
        desc="Each required piece of information includes at least one supporting URL",
        parent=src_node,
        critical=True,
    )

    # At least one official RRISD URL present among all citations
    evaluator.add_custom_node(
        result=any_official_rrisd_url(ex),
        id="at_least_one_official_rrisd_url",
        desc="At least one reference URL is from the official Round Rock ISD website",
        parent=src_node,
        critical=True,
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
    Evaluate an answer for the Round Rock ISD comprehensive (as of March 2026) task.
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
        default_model=model,
    )

    # Add a critical top-level node to mirror rubric's "Comprehensive..." root
    main = evaluator.add_parallel(
        id="comprehensive_main",
        desc="Provide all requested Round Rock ISD district and current superintendent information, time-qualified as of March 2026, with supporting URLs.",
        parent=root,
        critical=True,
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_rrisd(),
        template_class=RRISDExtraction,
        extraction_name="rrisd_extraction",
    )

    # Build verification subtrees
    await build_asof_node(evaluator, main, extracted)
    await build_district_info_nodes(evaluator, main, extracted)
    await build_superintendent_identity_education_nodes(evaluator, main, extracted)
    await build_superintendent_experience_tenure_nodes(evaluator, main, extracted)
    await build_compensation_node(evaluator, main, extracted)
    await build_sourcing_references_node(evaluator, main, extracted)

    # Return evaluation summary
    return evaluator.get_summary()