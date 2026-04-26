import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "career_center_director_2023_2024"
TASK_DESCRIPTION = """
Identify the name and current institution of a career center director in the United States who meets ALL of the following criteria:

Current Position:
- Assumed their position as Executive Director or Director of a university career center or career services office between January 2023 and December 2024 (inclusive)

Educational Background:
- Holds at least one advanced degree (Master's level or higher)
- Earned at least one degree (at any level) from an educational institution located on the African continent

International Work Experience:
- Has prior professional work experience outside the United States in at least two different countries
- This international work experience must have been in career development, career services, partnerships, student services, or directly related fields

Previous Employment:
- Immediately before their current university position, worked for an organization (foundation, NGO, or educational institution) whose primary mission focused on education or career development
- This previous organization operated programs or services in multiple countries

Professional Scope:
- In their previous international role, the person's programs or services reached or served over 1,000 students, employees, or beneficiaries
- Their responsibilities in this role spanned multiple geographic regions or countries

Required Information:
Provide the person's full name, their current position title, the name of their current institution, the month and year they assumed their current position, and reference URLs supporting each major criterion (current position, educational background, previous employment, and professional scope).
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Degree(BaseModel):
    degree_type: Optional[str] = None  # e.g., Master of Arts, MBA, PhD, EdD, M.Ed.
    field_of_study: Optional[str] = None
    institution_name: Optional[str] = None
    institution_country: Optional[str] = None
    graduation_year: Optional[str] = None


class InternationalRole(BaseModel):
    title: Optional[str] = None
    organization: Optional[str] = None
    countries: List[str] = Field(default_factory=list)  # countries worked in for this role
    field_area: Optional[str] = None  # career services, partnerships, student services, etc.
    responsibilities: Optional[str] = None
    timeframe: Optional[str] = None
    regions: List[str] = Field(default_factory=list)  # regions/continents (e.g., Sub-Saharan Africa, MENA)
    people_served_desc: Optional[str] = None  # e.g., "served 5,000 students"
    people_served_number: Optional[str] = None  # keep as string to be lenient
    urls: List[str] = Field(default_factory=list)


class PreviousEmployment(BaseModel):
    organization_name: Optional[str] = None
    organization_type: Optional[str] = None  # foundation, NGO, educational institution, etc.
    position_title: Optional[str] = None
    mission_description: Optional[str] = None
    countries_of_operation: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class BasicInfo(BaseModel):
    name: Optional[str] = None
    position_title: Optional[str] = None
    institution: Optional[str] = None
    appointment_month_year: Optional[str] = None  # e.g., "June 2023"
    urls: List[str] = Field(default_factory=list)  # reference(s) confirming basic info (bio/announcement page)


class CandidateExtraction(BaseModel):
    basic: Optional[BasicInfo] = None

    # Current position references (can repeat basic.urls if appropriate)
    position_urls: List[str] = Field(default_factory=list)

    # Educational background
    education_degrees: List[Degree] = Field(default_factory=list)
    education_urls: List[str] = Field(default_factory=list)

    # International work experience
    international_roles: List[InternationalRole] = Field(default_factory=list)
    international_urls: List[str] = Field(default_factory=list)

    # Immediate previous employment
    previous_employment: Optional[PreviousEmployment] = None

    # Professional scope (scale/scope) specific references (can overlap international/previous)
    professional_scope_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_candidate() -> str:
    return """
Extract the details for exactly one individual (the main candidate) who best matches ALL the criteria below from the provided answer. If multiple individuals are mentioned, pick the one that most closely satisfies the constraints; if ties remain, pick the first one presented. Return null for any field that is not explicitly available in the answer.

You must extract:
1) basic:
   - name: Full name of the person
   - position_title: Their current position title at the university (e.g., "Executive Director, Career Services")
   - institution: The current U.S. university or college
   - appointment_month_year: The month and year when they assumed their current position (e.g., "June 2023", "Oct 2024"). If only a year is stated (e.g., "2024"), extract that.
   - urls: A list of URLs that support the above basic details (bio, announcement, news release, etc.)

2) position_urls (array):
   - Any URLs that specifically confirm the current role details and start/announcement date

3) education_degrees (array of objects; include all degrees mentioned for this candidate):
   - degree_type (e.g., "Master of Arts", "MBA", "PhD", "EdD", "M.Ed.", "MS", "MA")
   - field_of_study (e.g., "Education Policy", "Career Counseling", "International Development")
   - institution_name (school/college/university name)
   - institution_country (country where the institution is located; if not explicit, but implied in the text, extract the country name)
   - graduation_year (if provided; otherwise null)

4) education_urls (array):
   - URLs supporting the candidate’s educational background

5) international_roles (array; include distinct international (non‑US) roles or assignments for the candidate):
   For each role:
   - title (job title or role name)
   - organization (employer/host organization)
   - countries (list of specific non-US countries where they worked or operated in this role)
   - field_area (e.g., career services, partnerships, student services, workforce development)
   - responsibilities (short description of responsibilities/duties)
   - timeframe (if provided; free text)
   - regions (list of geographic regions/continents, e.g., "Sub-Saharan Africa", "MENA", "Europe")
   - people_served_desc (verbatim snippet that mentions the scale, e.g., "served 5,000 students")
   - people_served_number (a number or threshold if present, e.g., "5000", "1000+"; otherwise null)
   - urls (URLs that substantiate this international role)

6) international_urls (array):
   - Additional URLs corroborating non‑US work experience (can overlap with above role urls)

7) previous_employment:
   - organization_name (the immediate previous organization before joining the current university)
   - organization_type (e.g., foundation, NGO, educational institution)
   - position_title (title held at the previous organization)
   - mission_description (the organization’s mission described in the answer)
   - countries_of_operation (array listing countries/regions where the org operates or has programs)
   - urls (URLs supporting this previous employment information)

8) professional_scope_urls (array):
   - URLs specifically supporting the scale/scope (e.g., over 1,000 served; multi-region responsibility; leadership role) in the previous international role.

General rules:
- Extract ONLY what is explicitly present in the answer. Do not invent URLs or details.
- When URLs are present in markdown format, extract the actual link targets.
- Allow reasonable variations in month names (e.g., "Oct 2024", "October 2024") for appointment_month_year.
- If a field is not stated, return null (or an empty list for arrays).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def combine_urls(*url_lists: Optional[List[str]]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if not isinstance(u, str):
                continue
            u2 = u.strip()
            if not u2:
                continue
            if u2 not in seen:
                seen.add(u2)
                result.append(u2)
    return result


def safe_str(s: Optional[str]) -> str:
    return s or ""


def pick_two_non_us_countries(countries: List[str]) -> List[str]:
    if not countries:
        return []
    us_aliases = {
        "united states", "united states of america", "usa", "u.s.a", "us", "u.s.", "america"
    }
    non_us = [c for c in countries if isinstance(c, str) and c.strip() and c.strip().lower() not in us_aliases]
    # De-duplicate preserving order
    seen = set()
    picked = []
    for c in non_us:
        c2 = c.strip()
        if c2.lower() not in seen:
            seen.add(c2.lower())
            picked.append(c2)
        if len(picked) >= 2:
            break
    return picked


def any_role_has_responsibilities(roles: List[InternationalRole]) -> bool:
    for r in roles:
        if r and r.responsibilities and r.responsibilities.strip():
            return True
    return False


def any_role_has_regions_or_countries(roles: List[InternationalRole]) -> bool:
    for r in roles:
        if (r.countries and len([c for c in r.countries if c and c.strip()])) or \
           (r.regions and len([reg for reg in r.regions if reg and reg.strip()])):
            return True
    return False


def any_role_has_people_scale(roles: List[InternationalRole]) -> bool:
    for r in roles:
        if (r.people_served_desc and r.people_served_desc.strip()) or \
           (r.people_served_number and r.people_served_number.strip()):
            return True
    return False


def select_african_degree_for_claim(degrees: List[Degree]) -> Optional[Degree]:
    """
    Heuristic: return the first degree that likely corresponds to an African institution.
    We do not hardcode the country list here; the verification LLM can use common knowledge to
    recognize whether the stated country is in Africa from the cited webpages.
    """
    if not degrees:
        return None
    for d in degrees:
        if (d.institution_country and d.institution_country.strip()) or (d.institution_name and d.institution_name.strip()):
            # Return the first degree that at least provides institution info;
            # LLM will verify Africa via the source content.
            return d
    return None


def collect_scope_urls(data: CandidateExtraction) -> List[str]:
    role_urls = []
    for r in data.international_roles:
        role_urls.extend(r.urls or [])
    prev_urls = data.previous_employment.urls if data.previous_employment else []
    return combine_urls(role_urls, data.international_urls, prev_urls, data.professional_scope_urls)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_basic_identification(evaluator: Evaluator, parent_node, data: CandidateExtraction) -> None:
    node = evaluator.add_parallel(
        id="Basic_Identification",
        desc="Verify basic identification information is provided",
        parent=parent_node,
        critical=False
    )

    b = data.basic or BasicInfo()

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=bool(b.name and b.name.strip()),
        id="Person_Name",
        desc="The full name of the individual is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(b.institution and b.institution.strip()),
        id="Current_Institution",
        desc="The name of the current US university where the person works is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(b.position_title and b.position_title.strip()),
        id="Position_Title",
        desc="The exact position title is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(b.appointment_month_year and b.appointment_month_year.strip()),
        id="Appointment_Date",
        desc="The month and year when the person assumed their current position is provided",
        parent=node,
        critical=True
    )

    # Reference URL supports basic info (critical)
    ref_leaf = evaluator.add_leaf(
        id="Basic_Info_Reference",
        desc="A reference URL supporting the basic identification information is provided",
        parent=node,
        critical=True
    )
    basic_sources = combine_urls(b.urls, data.position_urls)
    if not basic_sources:
        # Fail due to missing sources
        ref_leaf.score = 0.0
        ref_leaf.status = "failed"
    else:
        claim = (
            f"The page(s) confirm that {safe_str(b.name)} holds the position '{safe_str(b.position_title)}' "
            f"at {safe_str(b.institution)}, and indicate an appointment/announcement around '{safe_str(b.appointment_month_year)}'."
        )
        await evaluator.verify(
            claim=claim,
            node=ref_leaf,
            sources=basic_sources,
            additional_instruction=(
                "Check that the cited page(s) clearly mention the person's name, current title, and institution. "
                "Also look for a start or announcement date (month/year acceptable). Minor phrasing differences are acceptable."
            )
        )


async def verify_current_position(evaluator: Evaluator, parent_node, data: CandidateExtraction) -> None:
    node = evaluator.add_parallel(
        id="Current_Position_Verification",
        desc="Verify the current position meets all required criteria",
        parent=parent_node,
        critical=False
    )

    b = data.basic or BasicInfo()
    pos_sources = combine_urls(b.urls, data.position_urls)

    # US University Confirmation (critical)
    us_leaf = evaluator.add_leaf(
        id="US_University_Confirmation",
        desc="The position is at a university in the United States",
        parent=node,
        critical=True
    )
    if not pos_sources:
        us_leaf.score = 0.0
        us_leaf.status = "failed"
    else:
        claim = f"{safe_str(b.institution)} is a university or college located in the United States."
        await evaluator.verify(
            claim=claim,
            node=us_leaf,
            sources=pos_sources,
            additional_instruction=(
                "Use the page(s) to confirm the institution is a U.S. higher education institution. "
                "Signals include .edu domain, references to U.S. locations or accreditation. "
                "If the page clearly positions the institution in the U.S., consider it confirmed."
            )
        )

    # Director-Level Title (critical)
    dir_leaf = evaluator.add_leaf(
        id="Director_Level_Title",
        desc="The position title includes 'Executive Director' or 'Director'",
        parent=node,
        critical=True
    )
    if not pos_sources:
        dir_leaf.score = 0.0
        dir_leaf.status = "failed"
    else:
        claim = (
            f"The person's current position title is '{safe_str(b.position_title)}', which includes the word 'Director' "
            f"(or 'Executive Director'/'Director')."
        )
        await evaluator.verify(
            claim=claim,
            node=dir_leaf,
            sources=pos_sources,
            additional_instruction="Match word variants like 'Director', 'Executive Director', 'Director of Career Services', etc."
        )

    # Career Services Focus (critical)
    focus_leaf = evaluator.add_leaf(
        id="Career_Services_Focus",
        desc="The position is specifically for a career center, career services, or career development office",
        parent=node,
        critical=True
    )
    if not pos_sources:
        focus_leaf.score = 0.0
        focus_leaf.status = "failed"
    else:
        claim = (
            "The person's role is the director (or executive director) of a career center, career services, or career "
            "development office at the institution."
        )
        await evaluator.verify(
            claim=claim,
            node=focus_leaf,
            sources=pos_sources,
            additional_instruction="Look for phrases like 'Career Center', 'Career Services', 'Career Development', 'Career & Professional Development', etc."
        )

    # Appointment within timeframe (critical)
    timeframe_leaf = evaluator.add_leaf(
        id="Appointment_Within_Timeframe",
        desc="The person assumed their current position between January 2023 and December 2024 (inclusive)",
        parent=node,
        critical=True
    )
    if not pos_sources:
        timeframe_leaf.score = 0.0
        timeframe_leaf.status = "failed"
    else:
        claim = (
            f"The appointment/announcement for {safe_str(b.name)} taking the role occurred between January 2023 and "
            f"December 2024 (inclusive). The stated month/year is '{safe_str(b.appointment_month_year)}'."
        )
        await evaluator.verify(
            claim=claim,
            node=timeframe_leaf,
            sources=pos_sources,
            additional_instruction=(
                "Confirm that the page(s) show a start or announcement date within 2023-01 through 2024-12 inclusive. "
                "Month+year or just year (2023 or 2024) is acceptable as evidence."
            )
        )

    # Start Date Documented (non-critical)
    start_doc_leaf = evaluator.add_leaf(
        id="Start_Date_Documented",
        desc="The specific start date or announcement date is documented with evidence",
        parent=node,
        critical=False
    )
    if not pos_sources:
        start_doc_leaf.score = 0.0
        start_doc_leaf.status = "failed"
    else:
        claim = "The page(s) explicitly mention the start date or announcement date (at least month and year) for the appointment."
        await evaluator.verify(
            claim=claim,
            node=start_doc_leaf,
            sources=pos_sources,
            additional_instruction="Look for month and year; an exact day is not required."
        )

    # Position Reference (critical)
    ref_leaf = evaluator.add_leaf(
        id="Position_Reference",
        desc="A reference URL confirming the current position details and timeframe is provided",
        parent=node,
        critical=True
    )
    if not pos_sources:
        ref_leaf.score = 0.0
        ref_leaf.status = "failed"
    else:
        claim = (
            f"The page(s) confirm that {safe_str(b.name)} currently serves as '{safe_str(b.position_title)}' "
            f"at {safe_str(b.institution)} and include a start/announcement date in 2023 or 2024."
        )
        await evaluator.verify(
            claim=claim,
            node=ref_leaf,
            sources=pos_sources,
            additional_instruction="The page should indicate both the role and an associated date showing it began in 2023 or 2024."
        )


async def verify_education(evaluator: Evaluator, parent_node, data: CandidateExtraction) -> None:
    node = evaluator.add_parallel(
        id="Educational_Background_Verification",
        desc="Verify educational background meets all requirements",
        parent=parent_node,
        critical=False
    )

    degrees = data.education_degrees or []
    edu_sources = combine_urls(data.education_urls)

    # Advanced degree held (critical)
    adv_leaf = evaluator.add_leaf(
        id="Advanced_Degree_Held",
        desc="The person holds at least one Master's degree, PhD, or equivalent advanced degree",
        parent=node,
        critical=True
    )
    if not edu_sources:
        adv_leaf.score = 0.0
        adv_leaf.status = "failed"
    else:
        degree_types_list = [d.degree_type for d in degrees if d.degree_type]
        degree_types_str = ", ".join(degree_types_list) if degree_types_list else "an advanced degree (Master's or higher)"
        claim = f"The sources indicate the person holds at least one advanced degree (Master's or higher), e.g., {degree_types_str}."
        await evaluator.verify(
            claim=claim,
            node=adv_leaf,
            sources=edu_sources,
            additional_instruction="Accept common variants: MA, MS, MBA, M.Ed., MPA, MPP, MSW, MPhil, PhD, EdD, etc."
        )

    # Degree type specified (non-critical)
    evaluator.add_custom_node(
        result=any(d.degree_type and d.degree_type.strip() for d in degrees),
        id="Degree_Type_Specified",
        desc="The specific type(s) of advanced degree(s) are identified",
        parent=node,
        critical=False
    )

    # Field of study specified (non-critical)
    evaluator.add_custom_node(
        result=any(d.field_of_study and d.field_of_study.strip() for d in degrees),
        id="Field_Of_Study",
        desc="The field(s) of study for the advanced degree(s) are provided",
        parent=node,
        critical=False
    )

    # Degree from African institution (critical)
    african_leaf = evaluator.add_leaf(
        id="African_Institution_Degree",
        desc="At least one degree (at any level) was earned from an institution located on the African continent",
        parent=node,
        critical=True
    )
    if not edu_sources:
        african_leaf.score = 0.0
        african_leaf.status = "failed"
    else:
        afr_deg = select_african_degree_for_claim(degrees)
        if afr_deg:
            deg_label = afr_deg.degree_type or "a degree"
            inst = afr_deg.institution_name or "an institution"
            country = afr_deg.institution_country or "an African country"
            claim = f"The person earned {deg_label} from {inst} in {country}, which is located in Africa."
        else:
            claim = "The sources show that the person earned at least one degree from an educational institution located in an African country."
        await evaluator.verify(
            claim=claim,
            node=african_leaf,
            sources=edu_sources,
            additional_instruction="If the page confirms the institution and its country (e.g., Ghana, Kenya, Egypt), it is acceptable to recognize that country as being in Africa."
        )

    # African institution name provided (non-critical)
    evaluator.add_custom_node(
        result=any(d.institution_name and d.institution_name.strip() for d in degrees),
        id="African_Institution_Name",
        desc="The name of the African institution is provided",
        parent=node,
        critical=False
    )

    # African institution country provided (non-critical)
    evaluator.add_custom_node(
        result=any(d.institution_country and d.institution_country.strip() for d in degrees),
        id="African_Institution_Country",
        desc="The country where the African institution is located is specified",
        parent=node,
        critical=False
    )

    # Specific African degree identified (non-critical)
    evaluator.add_custom_node(
        result=any((d.degree_type and d.degree_type.strip()) and (d.institution_name and d.institution_name.strip()) for d in degrees),
        id="Specific_African_Degree",
        desc="The specific degree earned from the African institution is identified",
        parent=node,
        critical=False
    )

    # Educational background reference (critical)
    edu_ref_leaf = evaluator.add_leaf(
        id="Educational_Background_Reference",
        desc="A reference URL supporting the educational background information is provided",
        parent=node,
        critical=True
    )
    if not edu_sources:
        edu_ref_leaf.score = 0.0
        edu_ref_leaf.status = "failed"
    else:
        claim = "The page(s) support the person's educational background, including at least one advanced degree and the African institution degree."
        await evaluator.verify(
            claim=claim,
            node=edu_ref_leaf,
            sources=edu_sources,
            additional_instruction="Look for explicit listing of degrees, institutions, and any country context."
        )


async def verify_international_experience(evaluator: Evaluator, parent_node, data: CandidateExtraction) -> None:
    node = evaluator.add_parallel(
        id="International_Work_Experience_Verification",
        desc="Verify international work experience meets all requirements",
        parent=parent_node,
        critical=False
    )

    roles = data.international_roles or []
    all_countries = []
    all_role_urls = []
    for r in roles:
        if r.countries:
            all_countries.extend([c for c in r.countries if c and c.strip()])
        if r.urls:
            all_role_urls.extend(r.urls)
    intl_sources = combine_urls(all_role_urls, data.international_urls)

    # Non-US work experience (critical)
    non_us_leaf = evaluator.add_leaf(
        id="Non_US_Work_Experience",
        desc="The person has prior professional work experience outside the United States",
        parent=node,
        critical=True
    )
    if not intl_sources:
        non_us_leaf.score = 0.0
        non_us_leaf.status = "failed"
    else:
        showcase = ", ".join(pick_two_non_us_countries(all_countries)) or "at least one non-US country"
        claim = f"The person has prior professional work experience outside the United States (e.g., {showcase})."
        await evaluator.verify(
            claim=claim,
            node=non_us_leaf,
            sources=intl_sources,
            additional_instruction="Confirm the bio/resume pages indicate work outside the U.S."
        )

    # Two+ countries (critical)
    two_c_leaf = evaluator.add_leaf(
        id="Two_Plus_Countries",
        desc="The person has worked in at least two different countries outside the United States",
        parent=node,
        critical=True
    )
    if not intl_sources:
        two_c_leaf.score = 0.0
        two_c_leaf.status = "failed"
    else:
        examples = pick_two_non_us_countries(all_countries)
        ex_str = ", ".join(examples) if examples else "two or more non-US countries"
        claim = f"The person worked in at least two different non-US countries (e.g., {ex_str})."
        await evaluator.verify(
            claim=claim,
            node=two_c_leaf,
            sources=intl_sources,
            additional_instruction="The page(s) should clearly indicate two or more distinct non-US countries tied to the person's work."
        )

    # Career development-related field (critical)
    field_leaf = evaluator.add_leaf(
        id="Career_Development_Field",
        desc="The international work experience was in career development, career services, partnerships, student services, or directly related fields",
        parent=node,
        critical=True
    )
    if not intl_sources:
        field_leaf.score = 0.0
        field_leaf.status = "failed"
    else:
        claim = (
            "The international work experience was in career development/career services/partnerships/student services or closely related fields."
        )
        await evaluator.verify(
            claim=claim,
            node=field_leaf,
            sources=intl_sources,
            additional_instruction="Look for phrases explicitly indicating career services, student services, employability, workforce, or partnerships supporting career development."
        )

    # International positions listed (non-critical)
    evaluator.add_custom_node(
        result=any(r.title and r.title.strip() for r in roles),
        id="International_Positions_Listed",
        desc="Specific position title(s) held outside the US are provided",
        parent=node,
        critical=False
    )

    # Countries identified (non-critical)
    evaluator.add_custom_node(
        result=bool(all_countries),
        id="Countries_Identified",
        desc="The specific countries where the person worked are named",
        parent=node,
        critical=False
    )

    # Positions per country (non-critical)
    evaluator.add_custom_node(
        result=any((r.title and r.title.strip()) and (r.countries and len(r.countries) > 0) for r in roles),
        id="Positions_Per_Country",
        desc="The role or capacity in which the person worked in each country is described",
        parent=node,
        critical=False
    )

    # Responsibilities described (non-critical)
    evaluator.add_custom_node(
        result=any_role_has_responsibilities(roles),
        id="Responsibilities_Described",
        desc="The key responsibilities or duties in the international role(s) are described",
        parent=node,
        critical=False
    )

    # International experience reference (critical)
    intl_ref_leaf = evaluator.add_leaf(
        id="International_Experience_Reference",
        desc="A reference URL supporting the international work experience is provided",
        parent=node,
        critical=True
    )
    if not intl_sources:
        intl_ref_leaf.score = 0.0
        intl_ref_leaf.status = "failed"
    else:
        claim = "The page(s) substantiate the person's international (non-US) professional work experience."
        await evaluator.verify(
            claim=claim,
            node=intl_ref_leaf,
            sources=intl_sources,
            additional_instruction="Any bio, CV, or organization pages that list the person's international roles count."
        )


async def verify_previous_employment(evaluator: Evaluator, parent_node, data: CandidateExtraction) -> None:
    node = evaluator.add_parallel(
        id="Previous_Employment_Verification",
        desc="Verify previous employment immediately before current position meets requirements",
        parent=parent_node,
        critical=False
    )

    b = data.basic or BasicInfo()
    prev = data.previous_employment or PreviousEmployment()
    prev_sources = combine_urls(prev.urls)

    # Previous organization name (critical - existence)
    evaluator.add_custom_node(
        result=bool(prev.organization_name and prev.organization_name.strip()),
        id="Previous_Organization_Name",
        desc="The name of the previous organization is provided",
        parent=node,
        critical=True
    )

    # Organization type (non-critical - existence)
    evaluator.add_custom_node(
        result=bool(prev.organization_type and prev.organization_type.strip()),
        id="Organization_Type",
        desc="The type of organization (foundation, NGO, educational institution, etc.) is specified",
        parent=node,
        critical=False
    )

    # Position at previous organization (non-critical - existence)
    evaluator.add_custom_node(
        result=bool(prev.position_title and prev.position_title.strip()),
        id="Position_At_Previous_Organization",
        desc="The position title held at the previous organization is provided",
        parent=node,
        critical=False
    )

    # Education/Career mission (critical - verify)
    mission_leaf = evaluator.add_leaf(
        id="Education_Career_Mission",
        desc="The previous organization's primary mission or work focused on education, career development, student support, or scholarships",
        parent=node,
        critical=True
    )
    if not prev_sources:
        mission_leaf.score = 0.0
        mission_leaf.status = "failed"
    else:
        claim = (
            f"The organization '{safe_str(prev.organization_name)}' has a primary mission focused on education, "
            f"career development, student support, scholarships, or closely related aims."
        )
        await evaluator.verify(
            claim=claim,
            node=mission_leaf,
            sources=prev_sources,
            additional_instruction="Confirm via the organization's About/Mission pages or program descriptions."
        )

    # Multi-country operations (critical - verify)
    multi_leaf = evaluator.add_leaf(
        id="Multi_Country_Operations",
        desc="The previous organization operated programs or services in multiple countries (at least two)",
        parent=node,
        critical=True
    )
    if not prev_sources:
        multi_leaf.score = 0.0
        multi_leaf.status = "failed"
    else:
        claim = (
            f"The organization '{safe_str(prev.organization_name)}' operated programs or services in multiple countries (two or more)."
        )
        await evaluator.verify(
            claim=claim,
            node=multi_leaf,
            sources=prev_sources,
            additional_instruction="Accept explicit lists of multiple countries, statements like 'global presence across X countries', or 'operations in A and B'."
        )

    # Countries of operation listed (non-critical - existence)
    evaluator.add_custom_node(
        result=bool(prev.countries_of_operation and len(prev.countries_of_operation) > 0),
        id="Countries_Of_Operation_Listed",
        desc="Specific countries where the organization operated are named",
        parent=node,
        critical=False
    )

    # Mission description provided (non-critical - existence)
    evaluator.add_custom_node(
        result=bool(prev.mission_description and prev.mission_description.strip()),
        id="Mission_Description",
        desc="A description of the organization's mission or work is provided",
        parent=node,
        critical=False
    )

    # Previous employment reference (critical - verify immediate prior)
    prev_ref_leaf = evaluator.add_leaf(
        id="Previous_Employment_Reference",
        desc="A reference URL supporting the previous employment information is provided",
        parent=node,
        critical=True
    )
    mix_sources = combine_urls(prev_sources, (b.urls or []))
    if not mix_sources:
        prev_ref_leaf.score = 0.0
        prev_ref_leaf.status = "failed"
    else:
        claim = (
            f"The page(s) confirm that immediately before joining {safe_str(b.institution)}, "
            f"{safe_str(b.name)} worked at {safe_str(prev.organization_name)}."
        )
        await evaluator.verify(
            claim=claim,
            node=prev_ref_leaf,
            sources=mix_sources,
            additional_instruction="Look for language like 'prior to joining', 'immediately before', or chronology implying this was the directly previous position."
        )


async def verify_professional_scope(evaluator: Evaluator, parent_node, data: CandidateExtraction) -> None:
    node = evaluator.add_parallel(
        id="Professional_Scope_Verification",
        desc="Verify the scale and scope of previous international role",
        parent=parent_node,
        critical=False
    )

    roles = data.international_roles or []
    scope_sources = collect_scope_urls(data)

    # Served over 1,000 people (critical - verify)
    served_leaf = evaluator.add_leaf(
        id="Served_Over_1000_People",
        desc="During the previous international role, the person's programs or services reached or served over 1,000 students, employees, or beneficiaries",
        parent=node,
        critical=True
    )
    if not scope_sources:
        served_leaf.score = 0.0
        served_leaf.status = "failed"
    else:
        claim = (
            "In the previous international role, the person's programs or services reached over 1,000 students, employees, or beneficiaries."
        )
        await evaluator.verify(
            claim=claim,
            node=served_leaf,
            sources=scope_sources,
            additional_instruction="Accept phrasings like 'more than 1,000', '1k+', 'over 2,000', 'thousands'. If multiple numbers are presented across programs, confirm the scale exceeds 1,000."
        )

    # Multiple geographic regions (critical - verify)
    multi_geo_leaf = evaluator.add_leaf(
        id="Multiple_Geographic_Regions",
        desc="The person's responsibilities in the previous role spanned multiple geographic regions or countries",
        parent=node,
        critical=True
    )
    if not scope_sources:
        multi_geo_leaf.score = 0.0
        multi_geo_leaf.status = "failed"
    else:
        claim = "The person's responsibilities in the previous role spanned multiple regions or countries (more than one)."
        await evaluator.verify(
            claim=claim,
            node=multi_geo_leaf,
            sources=scope_sources,
            additional_instruction="Evidence can include multiple regions/continents listed, country lists, or language such as 'regional lead across A and B'."
        )

    # Specific numbers provided (non-critical - existence)
    evaluator.add_custom_node(
        result=any_role_has_people_scale(roles),
        id="Specific_Numbers_Provided",
        desc="Specific numbers or ranges indicating the scale of people served are provided",
        parent=node,
        critical=False
    )

    # Regions/countries listed (non-critical - existence)
    evaluator.add_custom_node(
        result=any_role_has_regions_or_countries(roles),
        id="Regions_Countries_Listed",
        desc="Specific regions, countries, or continents covered are listed",
        parent=node,
        critical=False
    )

    # Geographic scope description (non-critical - existence)
    evaluator.add_custom_node(
        result=any_role_has_regions_or_countries(roles) or any_role_has_responsibilities(roles),
        id="Geographic_Scope_Description",
        desc="Description of the geographic scope or reach of the programs is provided",
        parent=node,
        critical=False
    )

    # Leadership role (non-critical - verify)
    lead_leaf = evaluator.add_leaf(
        id="Leadership_Role",
        desc="The person held a leadership role in designing, executing, or managing professional development or career programs",
        parent=node,
        critical=False
    )
    if not scope_sources:
        lead_leaf.score = 0.0
        lead_leaf.status = "failed"
    else:
        claim = (
            "The person held a leadership role (e.g., led, directed, managed) in designing, executing, or managing professional development or career programs."
        )
        await evaluator.verify(
            claim=claim,
            node=lead_leaf,
            sources=scope_sources,
            additional_instruction="Accept indications such as 'led', 'directed', 'managed', 'oversaw', 'executive', 'head of', etc."
        )

    # Leadership responsibilities described (non-critical - existence)
    evaluator.add_custom_node(
        result=any_role_has_responsibilities(roles),
        id="Leadership_Responsibilities",
        desc="Specific leadership responsibilities are described",
        parent=node,
        critical=False
    )

    # Professional scope reference (critical - verify)
    scope_ref_leaf = evaluator.add_leaf(
        id="Professional_Scope_Reference",
        desc="A reference URL supporting the scale and scope information is provided",
        parent=node,
        critical=True
    )
    if not scope_sources:
        scope_ref_leaf.score = 0.0
        scope_ref_leaf.status = "failed"
    else:
        claim = "The page(s) substantiate the scale (over 1,000 served) and multi-region scope of the previous international role."
        await evaluator.verify(
            claim=claim,
            node=scope_ref_leaf,
            sources=scope_sources,
            additional_instruction="One or more pages must clearly support the scale and scope assertions."
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
    Evaluate an answer for the career center director identification task.
    """
    # Initialize evaluator with a sequential root to enforce logical ordering across major criteria
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Add a main container node for the rubric (non-critical to allow partial credit across criteria)
    main_node = evaluator.add_sequential(
        id="Career_Center_Director_Identification",
        desc="Verify that the identified individual meets all specified criteria for a career center director position",
        parent=root,
        critical=False
    )

    # 1) Extract structured data from the answer
    extracted: CandidateExtraction = await evaluator.extract(
        prompt=prompt_extract_candidate(),
        template_class=CandidateExtraction,
        extraction_name="candidate_extraction",
    )

    # 2) Build verification tree according to rubric
    await verify_basic_identification(evaluator, main_node, extracted)
    await verify_current_position(evaluator, main_node, extracted)
    await verify_education(evaluator, main_node, extracted)
    await verify_international_experience(evaluator, main_node, extracted)
    await verify_previous_employment(evaluator, main_node, extracted)
    await verify_professional_scope(evaluator, main_node, extracted)

    # 3) Return structured summary
    return evaluator.get_summary()