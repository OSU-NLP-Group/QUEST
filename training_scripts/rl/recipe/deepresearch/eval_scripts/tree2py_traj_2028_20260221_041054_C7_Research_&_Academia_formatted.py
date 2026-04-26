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
TASK_ID = "funded_phd_us_ai_cs"
TASK_DESCRIPTION = (
    "Identify a fully funded PhD program in Computer Science, Machine Learning, Artificial Intelligence, or Electrical "
    "Engineering & Computer Science at a major research university in the United States that meets the following requirements: "
    "(1) offers a minimum 5-year funding package that includes full tuition remission, an annual stipend, and health insurance coverage; "
    "(2) requires a doctoral dissertation committee of at least 4 members with at least 2 members from the student's graduate program or department; "
    "(3) has an application deadline between December 1 and December 31 for Fall admission; "
    "(4) requires a minimum undergraduate GPA of at least 3.5 on a 4.0 scale for competitive applicants; "
    "and (5) includes coursework, research, and teaching components in the degree requirements. "
    "For the identified program, provide the following information with supporting reference URLs: "
    "(a) the university name and specific program name, "
    "(b) the exact application deadline for Fall 2026 or Fall 2027 admission, "
    "(c) the stated minimum GPA requirement for competitive applicants, "
    "(d) the duration of the guaranteed funding package in years, "
    "(e) the annual stipend amount for PhD students for the most recent academic year available, "
    "(f) confirmation that full tuition is covered, "
    "(g) confirmation that health insurance is included, "
    "(h) the minimum number of dissertation committee members required, "
    "(i) the requirement for committee composition regarding program membership, "
    "(j) the minimum coursework requirement in units or courses, "
    "(k) the research requirement (e.g., dissertation, seminar participation), "
    "(l) the teaching requirement if any, "
    "(m) the typical time to degree completion, and "
    "(n) the required application materials."
)

ALLOWED_DISCIPLINES = [
    "Computer Science",
    "CS",
    "Machine Learning",
    "ML",
    "Artificial Intelligence",
    "AI",
    "Electrical Engineering & Computer Science",
    "EECS",
    "Electrical Engineering and Computer Science",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramCore(BaseModel):
    university_name: Optional[str] = None
    program_name: Optional[str] = None
    discipline: Optional[str] = None
    identity_urls: List[str] = Field(default_factory=list)


class FundingInfo(BaseModel):
    guaranteed_funding_years: Optional[str] = None  # e.g., "5 years", "at least 5 years"
    stipend_amount_annual: Optional[str] = None     # textual, e.g., "$38,000 (2025–26)"
    stipend_period_label: Optional[str] = None      # e.g., "2025–26", "AY 2025-2026"
    tuition_covered_statement: Optional[str] = None # e.g., "Full tuition is covered"
    health_insurance_statement: Optional[str] = None# e.g., "Health insurance included"
    funding_urls: List[str] = Field(default_factory=list)


class AdmissionsInfo(BaseModel):
    deadline_exact_date: Optional[str] = None       # e.g., "December 15, 2026"
    target_fall_year: Optional[str] = None          # "2026" or "2027"
    competitive_min_gpa: Optional[str] = None       # e.g., "3.5 on a 4.0 scale"
    admissions_urls: List[str] = Field(default_factory=list)


class CommitteeInfo(BaseModel):
    committee_min_members: Optional[str] = None     # e.g., "4"
    committee_composition_rule: Optional[str] = None# textual rule
    committee_urls: List[str] = Field(default_factory=list)


class DegreeRequirementsInfo(BaseModel):
    coursework_minimum: Optional[str] = None        # e.g., "48 units", "10 courses"
    research_requirement: Optional[str] = None      # textual, e.g., "dissertation", "research seminars"
    teaching_requirement: Optional[str] = None      # textual or "none"/"not specified"
    degree_urls: List[str] = Field(default_factory=list)


class OtherInfo(BaseModel):
    typical_time_to_degree: Optional[str] = None    # e.g., "5-6 years"
    application_materials: List[str] = Field(default_factory=list)
    other_urls: List[str] = Field(default_factory=list)


class ProgramExtraction(BaseModel):
    core: Optional[ProgramCore] = None
    funding: Optional[FundingInfo] = None
    admissions: Optional[AdmissionsInfo] = None
    committee: Optional[CommitteeInfo] = None
    degree: Optional[DegreeRequirementsInfo] = None
    other: Optional[OtherInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program() -> str:
    return (
        "Extract structured details for a single identified PhD program from the answer. "
        "Focus only on one program (the main one presented). "
        "Return fields exactly as mentioned in the answer; do not invent or infer. "
        "Additionally, extract official reference URLs (prefer .edu domains or official program pages) that support each category.\n\n"
        "Required JSON fields and descriptions:\n"
        "- core.university_name: The university name.\n"
        "- core.program_name: The specific PhD program name.\n"
        "- core.discipline: The field/discipline (e.g., Computer Science, Machine Learning, Artificial Intelligence, Electrical Engineering & Computer Science).\n"
        "- core.identity_urls: List of official URLs supporting the university/program identity.\n\n"
        "- funding.guaranteed_funding_years: The guaranteed funding duration (textual as given, e.g., '5 years', 'at least 5 years').\n"
        "- funding.stipend_amount_annual: The annual stipend amount for the most recent academic year available (textual as given).\n"
        "- funding.stipend_period_label: The year or effective period label associated with the stipend amount (textual as given).\n"
        "- funding.tuition_covered_statement: Text confirming full tuition coverage/remission.\n"
        "- funding.health_insurance_statement: Text confirming health insurance coverage.\n"
        "- funding.funding_urls: Official URLs supporting funding duration, stipend, tuition coverage, and health insurance.\n\n"
        "- admissions.deadline_exact_date: The exact application deadline for Fall 2026 or Fall 2027 admission (textual date as given, e.g., 'December 15, 2026').\n"
        "- admissions.target_fall_year: '2026' or '2027' depending on the deadline extracted.\n"
        "- admissions.competitive_min_gpa: The minimum undergraduate GPA requirement for competitive applicants (textual as given, e.g., '3.5 on a 4.0 scale').\n"
        "- admissions.admissions_urls: Official URLs supporting application deadline and GPA requirement.\n\n"
        "- committee.committee_min_members: Minimum number of dissertation committee members required (textual number as given).\n"
        "- committee.committee_composition_rule: Text describing composition rule (e.g., 'at least 2 members from the student's graduate program/department').\n"
        "- committee.committee_urls: Official URLs supporting committee size and composition.\n\n"
        "- degree.coursework_minimum: Minimum coursework requirement (units or courses; textual as given).\n"
        "- degree.research_requirement: Research requirement description (e.g., dissertation, seminars; textual as given).\n"
        "- degree.teaching_requirement: Teaching requirement description if any; otherwise 'none' or 'not specified' if the answer explicitly indicates no teaching requirement.\n"
        "- degree.degree_urls: Official URLs supporting coursework, research, and teaching requirements.\n\n"
        "- other.typical_time_to_degree: Typical time to degree completion (textual as given).\n"
        "- other.application_materials: List of required application materials (each item textual as given).\n"
        "- other.other_urls: Official URLs supporting typical time to degree and application materials.\n\n"
        "Rules:\n"
        "1) Extract only what is explicitly present in the answer; if missing, return null (or empty list for URLs/materials).\n"
        "2) For URLs fields, include only actual URLs that appear in the answer; prefer official sources (university/program pages, .edu domains).\n"
        "3) If the answer mentions both Fall 2026 and Fall 2027 deadlines, prefer Fall 2027; otherwise use whichever is present.\n"
        "4) Keep all values as strings where applicable; do not convert or compute numbers.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_text(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())

def _urls(u: Optional[List[str]]) -> List[str]:
    if not u:
        return []
    # Deduplicate and strip empties
    seen = set()
    clean: List[str] = []
    for x in u:
        if not x:
            continue
        s = x.strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            clean.append(s)
    return clean

def _discipline_claim(program_name: Optional[str], discipline: Optional[str]) -> str:
    pname = program_name or "the program"
    disc = discipline or "the stated discipline"
    return (
        f"The specified program '{pname}' is a Ph.D. in an allowed field "
        f"(Computer Science, Machine Learning, Artificial Intelligence, or Electrical Engineering & Computer Science). "
        f"In particular, it is described as '{disc}'."
    )

def _identity_us_research_claim(university_name: Optional[str], program_name: Optional[str]) -> str:
    uname = university_name or "the university"
    pname = program_name or "the program"
    return (
        f"The program '{pname}' is at '{uname}', which is a major research university located in the United States."
    )

def _funding_duration_claim(duration_text: Optional[str]) -> str:
    if _has_text(duration_text):
        return f"The guaranteed funding duration is '{duration_text}', and it is at least 5 years."
    return "The program guarantees a minimum funding duration of at least 5 years."

def _stipend_claim(amount_text: Optional[str], period_text: Optional[str]) -> str:
    if _has_text(amount_text) and _has_text(period_text):
        return f"The annual stipend amount for PhD students is '{amount_text}' for the period '{period_text}'."
    if _has_text(amount_text):
        return f"The annual stipend amount for PhD students is '{amount_text}'."
    return "The program provides an annual stipend amount for PhD students."

def _tuition_covered_claim() -> str:
    return "Full tuition is covered/remitted as part of the PhD funding package."

def _health_insurance_claim() -> str:
    return "Health insurance coverage is included as part of the PhD funding package."

def _deadline_claim(deadline_text: Optional[str], year_text: Optional[str]) -> str:
    yr = (year_text or "").strip()
    if _has_text(deadline_text) and yr in {"2026", "2027"}:
        return (
            f"The exact application deadline for Fall {yr} admission is '{deadline_text}', "
            "and the date falls between December 1 and December 31."
        )
    # General fallback
    return (
        "The exact application deadline for Fall 2026 or Fall 2027 admission falls between December 1 and December 31."
    )

def _gpa_claim(gpa_text: Optional[str]) -> str:
    if _has_text(gpa_text):
        return (
            f"The stated minimum undergraduate GPA requirement for competitive applicants is '{gpa_text}', "
            "which is at least 3.5 on a 4.0 scale."
        )
    return "The stated minimum undergraduate GPA requirement for competitive applicants is at least 3.5 on a 4.0 scale."

def _committee_min_claim(min_members_text: Optional[str]) -> str:
    if _has_text(min_members_text):
        return f"The minimum number of dissertation committee members required is '{min_members_text}', and it is at least 4."
    return "The minimum number of dissertation committee members required is at least 4."

def _committee_comp_rule_claim(rule_text: Optional[str]) -> str:
    if _has_text(rule_text):
        return (
            f"The committee composition rule specifies at least 2 members from the student's graduate program or department; "
            f"specifically: '{rule_text}'."
        )
    return "The committee composition rule specifies at least 2 members from the student's graduate program or department."

def _coursework_claim(text: Optional[str]) -> str:
    if _has_text(text):
        return f"The minimum coursework requirement is specified as '{text}'."
    return "A minimum coursework requirement (units or courses) is specified."

def _research_claim(text: Optional[str]) -> str:
    if _has_text(text):
        return f"The program's research requirement is described as '{text}'."
    return "The program requires research components (e.g., dissertation, research seminars)."

def _teaching_claim(text: Optional[str]) -> str:
    if _has_text(text):
        val = text.strip().lower()
        if val in {"none", "no", "not specified"}:
            return "No formal teaching requirement is specified by the program."
        return f"The program includes a teaching requirement described as '{text}'."
    return "No formal teaching requirement is specified by the program."

def _time_to_degree_claim(text: Optional[str]) -> str:
    if _has_text(text):
        return f"The typical time to degree completion is '{text}'."
    return "The program provides a typical time to degree completion."

def _materials_claim(materials: List[str]) -> str:
    if materials:
        return f"The required application materials include {materials}."
    return "The required application materials are listed on the official pages."

def _fail_if_no_urls_instruction(urls: List[str]) -> str:
    return (
        "You must return 'Incorrect' if there are no URLs provided for this verification. "
        "If URLs are provided, judge based on the evidence in the URLs."
        + ("" if urls else " (No URLs provided here.)")
    )

# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_identity_and_eligibility(evaluator: Evaluator, parent, data: ProgramExtraction):
    node = evaluator.add_parallel(
        id="program_identity_and_eligibility",
        desc="Program identity and baseline eligibility constraints are satisfied",
        parent=parent,
        critical=True
    )

    core = data.core or ProgramCore()
    id_urls = _urls(core.identity_urls)

    # University name provided (existence check)
    evaluator.add_custom_node(
        result=_has_text(core.university_name),
        id="university_name_provided",
        desc="University name is provided",
        parent=node,
        critical=True
    )

    # Program name provided (existence check)
    evaluator.add_custom_node(
        result=_has_text(core.program_name),
        id="program_name_provided",
        desc="Specific PhD program name is provided",
        parent=node,
        critical=True
    )

    # Discipline in allowed fields (verification via identity URLs)
    disc_leaf = evaluator.add_leaf(
        id="discipline_in_allowed_fields",
        desc="Program is a PhD in Computer Science, Machine Learning, Artificial Intelligence, or Electrical Engineering & Computer Science",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=_discipline_claim(core.program_name, core.discipline),
        node=disc_leaf,
        sources=id_urls,
        additional_instruction=(
            "Check the official program page to confirm the discipline is one of: Computer Science, Machine Learning, "
            "Artificial Intelligence, or Electrical Engineering & Computer Science. Consider common abbreviations (CS, ML, AI, EECS) "
            "and exact program naming. " + _fail_if_no_urls_instruction(id_urls)
        )
    )

    # US research university (verification via identity URLs)
    us_leaf = evaluator.add_leaf(
        id="us_research_university",
        desc="Program is at a research university in the United States",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=_identity_us_research_claim(core.university_name, core.program_name),
        node=us_leaf,
        sources=id_urls,
        additional_instruction=(
            "Confirm the university is located in the United States and is a recognized research university. "
            "Evidence can include the university 'About' page, location, or program overview. "
            + _fail_if_no_urls_instruction(id_urls)
        )
    )


async def verify_funding_package(evaluator: Evaluator, parent, data: ProgramExtraction):
    node = evaluator.add_parallel(
        id="funding_package",
        desc="Funding package meets the fully-funded and minimum-duration constraints, and requested stipend detail is provided",
        parent=parent,
        critical=True
    )

    funding = data.funding or FundingInfo()
    f_urls = _urls(funding.funding_urls)

    # Guaranteed funding duration >= 5 years
    dur_leaf = evaluator.add_leaf(
        id="guaranteed_funding_duration_gte_5_years",
        desc="Guaranteed funding duration (in years) is provided and is at least 5 years",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=_funding_duration_claim(funding.guaranteed_funding_years),
        node=dur_leaf,
        sources=f_urls,
        additional_instruction=(
            "Confirm the page states a guaranteed funding duration and that it is at least 5 years. "
            + _fail_if_no_urls_instruction(f_urls)
        )
    )

    # Annual stipend amount provided (most recent academic year)
    stipend_leaf = evaluator.add_leaf(
        id="annual_stipend_amount_provided",
        desc="Annual stipend amount is provided for the most recent academic year available (with the year/effective period stated)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=_stipend_claim(funding.stipend_amount_annual, funding.stipend_period_label),
        node=stipend_leaf,
        sources=f_urls,
        additional_instruction=(
            "Verify that the official page provides an annual stipend amount for PhD students and, if available, the corresponding academic year/period label. "
            "Accept reasonable variations (e.g., 9-month stipend explicitly stated). "
            + _fail_if_no_urls_instruction(f_urls)
        )
    )

    # Full tuition covered
    tuition_leaf = evaluator.add_leaf(
        id="full_tuition_covered",
        desc="It is explicitly confirmed that full tuition is covered/remitted as part of the funding",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=_tuition_covered_claim(),
        node=tuition_leaf,
        sources=f_urls,
        additional_instruction=(
            "Confirm the official page explicitly states full tuition is covered or remitted. "
            + _fail_if_no_urls_instruction(f_urls)
        )
    )

    # Health insurance included
    health_leaf = evaluator.add_leaf(
        id="health_insurance_included",
        desc="It is explicitly confirmed that health insurance coverage is included as part of the funding",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=_health_insurance_claim(),
        node=health_leaf,
        sources=f_urls,
        additional_instruction=(
            "Confirm the official page explicitly states health insurance coverage is included (or substantially equivalent phrasing). "
            + _fail_if_no_urls_instruction(f_urls)
        )
    )


async def verify_admissions_constraints(evaluator: Evaluator, parent, data: ProgramExtraction):
    node = evaluator.add_parallel(
        id="admissions_constraints",
        desc="Admissions deadline and competitive GPA constraints are satisfied",
        parent=parent,
        critical=True
    )

    adm = data.admissions or AdmissionsInfo()
    a_urls = _urls(adm.admissions_urls)

    # Fall 2026 or 2027 exact deadline in range
    deadline_leaf = evaluator.add_leaf(
        id="fall_2026_or_2027_deadline_exact_and_in_range",
        desc="The exact application deadline is given for Fall 2026 or Fall 2027 admission and the date falls between December 1 and December 31",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=_deadline_claim(adm.deadline_exact_date, adm.target_fall_year),
        node=deadline_leaf,
        sources=a_urls,
        additional_instruction=(
            "Check that the deadline is for Fall 2026 or Fall 2027 admission and that it is between December 1 and December 31. "
            "If the extracted year is not 2026 or 2027, or the date is outside December 1–31, return Incorrect. "
            + _fail_if_no_urls_instruction(a_urls)
        )
    )

    # Competitive minimum GPA >= 3.5
    gpa_leaf = evaluator.add_leaf(
        id="competitive_min_gpa_gte_3_5",
        desc="The stated minimum undergraduate GPA requirement for competitive applicants is provided and is at least 3.5 on a 4.0 scale",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=_gpa_claim(adm.competitive_min_gpa),
        node=gpa_leaf,
        sources=a_urls,
        additional_instruction=(
            "Verify the official page states a minimum GPA for competitive applicants and that it is at least 3.5 on a 4.0 scale. "
            + _fail_if_no_urls_instruction(a_urls)
        )
    )


async def verify_committee_requirements(evaluator: Evaluator, parent, data: ProgramExtraction):
    node = evaluator.add_parallel(
        id="dissertation_committee_requirements",
        desc="Dissertation committee size and composition constraints are satisfied",
        parent=parent,
        critical=True
    )

    com = data.committee or CommitteeInfo()
    c_urls = _urls(com.committee_urls)

    # Minimum dissertation committee members >= 4
    members_leaf = evaluator.add_leaf(
        id="committee_min_members_gte_4",
        desc="Minimum number of dissertation committee members is provided and is at least 4",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=_committee_min_claim(com.committee_min_members),
        node=members_leaf,
        sources=c_urls,
        additional_instruction=(
            "Confirm the official page states the minimum committee size and that it is at least 4. "
            + _fail_if_no_urls_instruction(c_urls)
        )
    )

    # Composition rule: at least 2 from program/department
    comp_leaf = evaluator.add_leaf(
        id="committee_composition_min_2_from_program",
        desc="Committee composition rule is provided and specifies at least 2 members from the student's graduate program or department",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=_committee_comp_rule_claim(com.committee_composition_rule),
        node=comp_leaf,
        sources=c_urls,
        additional_instruction=(
            "Verify that the composition rule requires at least 2 committee members from the student's graduate program/department. "
            + _fail_if_no_urls_instruction(c_urls)
        )
    )


async def verify_degree_requirements(evaluator: Evaluator, parent, data: ProgramExtraction):
    node = evaluator.add_parallel(
        id="degree_requirement_components",
        desc="Degree requirements/components are provided (coursework and research required; teaching described if applicable)",
        parent=parent,
        critical=True
    )

    deg = data.degree or DegreeRequirementsInfo()
    d_urls = _urls(deg.degree_urls)

    # Coursework minimum specified
    coursework_leaf = evaluator.add_leaf(
        id="coursework_minimum_specified",
        desc="Minimum coursework requirement is provided in units or courses",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=_coursework_claim(deg.coursework_minimum),
        node=coursework_leaf,
        sources=d_urls,
        additional_instruction=(
            "Confirm the official page specifies a minimum coursework requirement (units or courses). "
            + _fail_if_no_urls_instruction(d_urls)
        )
    )

    # Research requirement specified
    research_leaf = evaluator.add_leaf(
        id="research_requirement_specified",
        desc="Research requirement is described (e.g., dissertation and any required research seminars/participation if stated)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=_research_claim(deg.research_requirement),
        node=research_leaf,
        sources=d_urls,
        additional_instruction=(
            "Verify the official page describes research requirements (e.g., dissertation, research seminars). "
            + _fail_if_no_urls_instruction(d_urls)
        )
    )

    # Teaching requirement described if any (set to critical True to satisfy tree constraints)
    teaching_leaf = evaluator.add_leaf(
        id="teaching_requirement_described_if_any",
        desc="Teaching requirement is described if the program has one; otherwise the answer clearly states that no teaching requirement is specified",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=_teaching_claim(deg.teaching_requirement),
        node=teaching_leaf,
        sources=d_urls,
        additional_instruction=(
            "If a teaching requirement exists, confirm it on the official page. "
            "If none is specified and the answer explicitly claims none, accept if the official pages do not indicate a requirement. "
            + _fail_if_no_urls_instruction(d_urls)
        )
    )


async def verify_other_requested_information(evaluator: Evaluator, parent, data: ProgramExtraction):
    node = evaluator.add_parallel(
        id="required_additional_requested_information",
        desc="Other explicitly requested program details are provided",
        parent=parent,
        critical=True
    )

    other = data.other or OtherInfo()
    o_urls = _urls(other.other_urls)
    # It is common that application materials also appear on admissions pages; include those too for robustness.
    adm_urls = _urls((data.admissions or AdmissionsInfo()).admissions_urls)
    combined_materials_urls = _urls(o_urls + adm_urls)

    # Typical time to degree provided
    time_leaf = evaluator.add_leaf(
        id="typical_time_to_degree_provided",
        desc="Typical time to degree completion is provided",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=_time_to_degree_claim(other.typical_time_to_degree),
        node=time_leaf,
        sources=o_urls,
        additional_instruction=(
            "Confirm the typical time to degree completion on official sources. "
            + _fail_if_no_urls_instruction(o_urls)
        )
    )

    # Application materials listed
    materials_leaf = evaluator.add_leaf(
        id="application_materials_listed",
        desc="Required application materials are listed",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=_materials_claim(other.application_materials),
        node=materials_leaf,
        sources=combined_materials_urls,
        additional_instruction=(
            "Verify that the official pages list the required application materials. "
            + _fail_if_no_urls_instruction(combined_materials_urls)
        )
    )


async def verify_official_urls(evaluator: Evaluator, parent, data: ProgramExtraction):
    node = evaluator.add_parallel(
        id="official_source_urls_provided",
        desc="Supporting reference URLs from official university/program sources are provided for each required claim/attribute",
        parent=parent,
        critical=True
    )

    core = data.core or ProgramCore()
    funding = data.funding or FundingInfo()
    adm = data.admissions or AdmissionsInfo()
    com = data.committee or CommitteeInfo()
    deg = data.degree or DegreeRequirementsInfo()
    other = data.other or OtherInfo()

    id_urls = _urls(core.identity_urls)
    f_urls = _urls(funding.funding_urls)
    a_urls = _urls(adm.admissions_urls)
    c_urls = _urls(com.committee_urls)
    d_urls = _urls(deg.degree_urls)
    o_urls = _urls(other.other_urls)

    # Identity URLs support identity claims
    urls_identity_leaf = evaluator.add_leaf(
        id="urls_support_identity",
        desc="Official URL(s) support the university/program identity claims",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided URLs are official university/program pages that clearly identify the university and the PhD program.",
        node=urls_identity_leaf,
        sources=id_urls,
        additional_instruction=(
            "Return Incorrect if there are no URLs. Prefer .edu domains or official program pages. "
            "Judge based on whether the pages clearly identify the university and program."
            + (" (No URLs provided.)" if not id_urls else "")
        )
    )

    # Funding URLs support funding claims
    urls_funding_leaf = evaluator.add_leaf(
        id="urls_support_funding",
        desc="Official URL(s) support the funding package claims (duration, stipend, tuition coverage, health insurance)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided URLs are official pages that state funding duration, stipend, tuition coverage, and health insurance.",
        node=urls_funding_leaf,
        sources=f_urls,
        additional_instruction=(
            "Return Incorrect if there are no URLs. Verify that the pages explicitly mention each of: funding duration, stipend, tuition coverage, and health insurance."
            + (" (No URLs provided.)" if not f_urls else "")
        )
    )

    # Admissions URLs support admissions claims
    urls_adm_leaf = evaluator.add_leaf(
        id="urls_support_admissions",
        desc="Official URL(s) support the admissions claims (deadline window/date, competitive minimum GPA)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided URLs are official pages that state the application deadline and the minimum GPA for competitive applicants.",
        node=urls_adm_leaf,
        sources=a_urls,
        additional_instruction=(
            "Return Incorrect if there are no URLs. Verify that the pages show the exact deadline date and the minimum GPA."
            + (" (No URLs provided.)" if not a_urls else "")
        )
    )

    # Committee URLs support committee claims
    urls_comm_leaf = evaluator.add_leaf(
        id="urls_support_committee",
        desc="Official URL(s) support the dissertation committee size and composition claims",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided URLs are official pages that state the minimum committee size and the composition rule (at least 2 members from the program/department).",
        node=urls_comm_leaf,
        sources=c_urls,
        additional_instruction=(
            "Return Incorrect if there are no URLs. Verify that the pages provide both size and composition requirements."
            + (" (No URLs provided.)" if not c_urls else "")
        )
    )

    # Degree URLs support degree requirement claims
    urls_degree_leaf = evaluator.add_leaf(
        id="urls_support_degree_requirements",
        desc="Official URL(s) support the coursework and research requirement claims (and teaching requirement if one is claimed)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided URLs are official pages that state the minimum coursework and research requirements, and any teaching requirement if present.",
        node=urls_degree_leaf,
        sources=d_urls,
        additional_instruction=(
            "Return Incorrect if there are no URLs. Verify that the pages provide coursework minimum, research components, and teaching requirement if applicable."
            + (" (No URLs provided.)" if not d_urls else "")
        )
    )

    # Other URLs support time to degree and application materials
    urls_other_leaf = evaluator.add_leaf(
        id="urls_support_other_requested_details",
        desc="Official URL(s) support typical time to degree and required application materials",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided URLs are official pages that state the typical time to degree and list the required application materials.",
        node=urls_other_leaf,
        sources=o_urls,
        additional_instruction=(
            "Return Incorrect if there are no URLs. Verify that the pages include time to degree information and a materials list."
            + (" (No URLs provided.)" if not o_urls else "")
        )
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
    Evaluate an answer for the fully funded US PhD program task and return a structured result.
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

    # Extract all structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_program(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction"
    )

    # Optional: record allowed disciplines for context in summary
    evaluator.add_custom_info(
        info={"allowed_disciplines": ALLOWED_DISCIPLINES},
        info_type="constraints",
        info_name="allowed_fields"
    )

    # Build and verify subtrees
    await verify_identity_and_eligibility(evaluator, root, extracted)
    await verify_funding_package(evaluator, root, extracted)
    await verify_admissions_constraints(evaluator, root, extracted)
    await verify_committee_requirements(evaluator, root, extracted)
    await verify_degree_requirements(evaluator, root, extracted)
    await verify_other_requested_information(evaluator, root, extracted)
    await verify_official_urls(evaluator, root, extracted)

    # Return the final summary
    return evaluator.get_summary()