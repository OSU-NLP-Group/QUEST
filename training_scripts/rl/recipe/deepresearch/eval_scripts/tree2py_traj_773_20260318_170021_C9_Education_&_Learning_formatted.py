import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "public_universities_eng_arenas_supports"
TASK_DESCRIPTION = """
I am researching public universities in the United States that combine strong engineering programs with comprehensive campus facilities and student support services. I need to find three public universities that meet all of the following criteria:

1. The university must offer at least one ABET-accredited undergraduate engineering program.
2. The university must have an on-campus basketball arena with a seating capacity of at least 20,000.
3. The university must have a student-to-faculty ratio of 15:1 or better (meaning 15 students or fewer per faculty member).
4. The university must have a four-year graduation rate of at least 45% for bachelor's degree students.
5. The university must offer an honors program that is open to incoming freshmen, with publicly stated admission criteria that include minimum standardized test scores (SAT or ACT) and/or minimum high school GPA requirements.
6. The university must offer study abroad programs that are available to undergraduate students.
7. The university must have on-campus recreation or fitness facilities available to students.
8. The university must provide career services that include internship or job placement support for students.
9. The university must have on-campus dining facilities with meal plan options available to students.
10. The university must provide campus shuttle or transportation services for students.
11. The university must have campus safety or security services that operate 24 hours a day, 7 days a week.

For each of the three universities, provide the following information:
- University name and location (city, state)
- Name of at least one ABET-accredited engineering program
- Name and seating capacity of the on-campus basketball arena
- Student-to-faculty ratio
- Four-year graduation rate (as a percentage)
- Honors program name and admission criteria (including any minimum SAT/ACT scores or GPA requirements stated)
- Confirmation that study abroad programs, recreation facilities, career services, dining facilities, transportation services, and 24/7 campus safety services are available
- Reference URLs for each piece of information provided
"""


# -----------------------------------------------------------------------------
# Extraction data models
# -----------------------------------------------------------------------------
class AbetInfo(BaseModel):
    program_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ArenaInfo(BaseModel):
    name: Optional[str] = None
    capacity: Optional[str] = None  # Keep as string to be robust to ranges/approx.
    sources: List[str] = Field(default_factory=list)


class RatioInfo(BaseModel):
    ratio: Optional[str] = None  # e.g., "14:1", "15 to 1"
    sources: List[str] = Field(default_factory=list)


class GraduationInfo(BaseModel):
    four_year_rate: Optional[str] = None  # e.g., "48%", "0.48", "about 50%"
    sources: List[str] = Field(default_factory=list)


class HonorsCriteria(BaseModel):
    program_name: Optional[str] = None
    accepts_freshmen: Optional[bool] = None
    min_sat: Optional[str] = None
    min_act: Optional[str] = None
    min_gpa: Optional[str] = None
    criteria_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class StudyAbroadInfo(BaseModel):
    description: Optional[str] = None
    undergrad_available: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class RecreationInfo(BaseModel):
    description: Optional[str] = None
    student_access: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class CareerInfo(BaseModel):
    description: Optional[str] = None
    internship_or_job_support: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class DiningInfo(BaseModel):
    description: Optional[str] = None
    meal_plans: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class TransportInfo(BaseModel):
    description: Optional[str] = None
    student_access: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class SafetyInfo(BaseModel):
    description: Optional[str] = None
    operates_24_7: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class PublicInstitutionInfo(BaseModel):
    is_public: Optional[bool] = None
    is_in_us: Optional[bool] = None
    location: Optional[str] = None  # "City, State" if available
    sources: List[str] = Field(default_factory=list)


class University(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None

    abet: Optional[AbetInfo] = None
    arena: Optional[ArenaInfo] = None
    ratio: Optional[RatioInfo] = None
    grad_rate: Optional[GraduationInfo] = None
    honors: Optional[HonorsCriteria] = None
    study_abroad: Optional[StudyAbroadInfo] = None
    recreation: Optional[RecreationInfo] = None
    career: Optional[CareerInfo] = None
    dining: Optional[DiningInfo] = None
    transport: Optional[TransportInfo] = None
    safety: Optional[SafetyInfo] = None
    institution: Optional[PublicInstitutionInfo] = None


class UniversitiesExtraction(BaseModel):
    universities: List[University] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_universities() -> str:
    return """
Extract up to the first three distinct public universities described in the answer, and structure the information exactly as the JSON schema below. Follow all rules carefully.

GENERAL RULES:
- Extract only what is explicitly presented in the answer text.
- Use strings for numeric-looking fields (e.g., capacities, ratios, percentages) to allow ranges or approximations.
- For each category, collect all explicit URLs cited in the answer that support that item. Do not invent URLs. If none are cited, return an empty list for 'sources'.
- If a field is not mentioned, set it to null (or [] for lists).
- Booleans should be true/false only if the answer explicitly states so; otherwise null.

REQUIRED OUTPUT SCHEMA (return exactly this shape):
{
  "universities": [
    {
      "name": null,
      "city": null,
      "state": null,
      "abet": {
        "program_name": null,
        "sources": []
      },
      "arena": {
        "name": null,
        "capacity": null,
        "sources": []
      },
      "ratio": {
        "ratio": null,
        "sources": []
      },
      "grad_rate": {
        "four_year_rate": null,
        "sources": []
      },
      "honors": {
        "program_name": null,
        "accepts_freshmen": null,
        "min_sat": null,
        "min_act": null,
        "min_gpa": null,
        "criteria_text": null,
        "sources": []
      },
      "study_abroad": {
        "description": null,
        "undergrad_available": null,
        "sources": []
      },
      "recreation": {
        "description": null,
        "student_access": null,
        "sources": []
      },
      "career": {
        "description": null,
        "internship_or_job_support": null,
        "sources": []
      },
      "dining": {
        "description": null,
        "meal_plans": null,
        "sources": []
      },
      "transport": {
        "description": null,
        "student_access": null,
        "sources": []
      },
      "safety": {
        "description": null,
        "operates_24_7": null,
        "sources": []
      },
      "institution": {
        "is_public": null,
        "is_in_us": null,
        "location": null,
        "sources": []
      }
    }
  ]
}

SPECIAL URL RULES:
- Only include URLs explicitly present in the answer (plain URLs or Markdown links).
- If a URL lacks protocol, prepend http://.
- Do not include non-URL references.

Limit to the first three universities mentioned; if fewer than three are present, return the available ones only.
    """.strip()


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _nz(s: Optional[str]) -> str:
    return s or ""


def _safe_sources(srcs: Optional[List[str]]) -> List[str]:
    return srcs or []


def _uni_label(idx: int) -> str:
    return ["First", "Second", "Third"][idx] if idx < 3 else f"University #{idx+1}"


# -----------------------------------------------------------------------------
# Verification builder for a single university
# -----------------------------------------------------------------------------
async def verify_university(evaluator: Evaluator, parent_node, uni: University, idx: int) -> None:
    uni_name = _nz(uni.name)
    city = _nz(uni.city)
    state = _nz(uni.state)
    unode = evaluator.add_parallel(
        id=f"university_{idx+1}",
        desc=f"{_uni_label(idx)} university meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # ------------------------- Engineering Accreditation ----------------------
    eng_node = evaluator.add_parallel(
        id=f"u{idx+1}_engineering_accreditation",
        desc="University offers at least one ABET-accredited undergraduate engineering program",
        parent=unode,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(uni.abet and _nz(uni.abet.program_name).strip()),
        id=f"u{idx+1}_program_exists",
        desc="At least one ABET-accredited engineering program is identified",
        parent=eng_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(uni.abet and len(_safe_sources(uni.abet.sources)) > 0),
        id=f"u{idx+1}_reference_url_engineering",
        desc="Reference URL provided for engineering program accreditation",
        parent=eng_node,
        critical=True
    )
    abet_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_abet_verification",
        desc="ABET accreditation is verified through official source",
        parent=eng_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided source is an official ABET accreditation listing that shows at least one accredited undergraduate engineering program at {uni_name}.",
        node=abet_leaf,
        sources=_safe_sources(uni.abet.sources if uni.abet else []),
        additional_instruction="Only pass if the page is clearly an official ABET accreditation resource (e.g., abet.org domain or the ABET Accredited Program Search) explicitly listing an accredited undergraduate engineering program for the named university. University department pages or third-party summaries are insufficient."
    )

    # ------------------------- Basketball Arena -------------------------------
    arena_node = evaluator.add_parallel(
        id=f"u{idx+1}_basketball_arena",
        desc="University has an on-campus basketball arena with seating capacity of at least 20,000",
        parent=unode,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(uni.arena and _nz(uni.arena.name).strip()),
        id=f"u{idx+1}_arena_exists",
        desc="On-campus basketball arena is identified",
        parent=arena_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(uni.arena and len(_safe_sources(uni.arena.sources)) > 0),
        id=f"u{idx+1}_reference_url_arena",
        desc="Reference URL provided for arena capacity information",
        parent=arena_node,
        critical=True
    )
    arena_cap_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_capacity_meets_minimum",
        desc="Arena seating capacity is 20,000 or greater",
        parent=arena_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The on-campus basketball arena {_nz(uni.arena.name if uni.arena else '')} at {uni_name} has a basketball seating capacity of at least 20,000.",
        node=arena_cap_leaf,
        sources=_safe_sources(uni.arena.sources if uni.arena else []),
        additional_instruction="Check the stated basketball seating capacity. If multiple capacities are listed (event vs. basketball), evaluate basketball capacity. Accept phrasing like 'approx. 21,000'. Fail if clearly below 20,000."
    )

    # ------------------------- Student-Faculty Ratio --------------------------
    ratio_node = evaluator.add_parallel(
        id=f"u{idx+1}_student_faculty_ratio",
        desc="University has a student-to-faculty ratio of 15:1 or better",
        parent=unode,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(uni.ratio and _nz(uni.ratio.ratio).strip()),
        id=f"u{idx+1}_ratio_identified",
        desc="Student-to-faculty ratio is identified",
        parent=ratio_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(uni.ratio and len(_safe_sources(uni.ratio.sources)) > 0),
        id=f"u{idx+1}_reference_url_ratio",
        desc="Reference URL provided for student-faculty ratio",
        parent=ratio_node,
        critical=True
    )
    ratio_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_ratio_meets_threshold",
        desc="Ratio is 15:1 or lower (better)",
        parent=ratio_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The student-to-faculty ratio at {uni_name} is 15:1 or better (15 or fewer students per faculty member).",
        node=ratio_leaf,
        sources=_safe_sources(uni.ratio.sources if uni.ratio else []),
        additional_instruction="Accept textual variants such as '15:1', '15 to 1', or 'fourteen to one' etc. Fail if the page clearly indicates a ratio worse than 15:1 (e.g., 16:1 or higher)."
    )

    # ------------------------- Graduation Rate --------------------------------
    grad_node = evaluator.add_parallel(
        id=f"u{idx+1}_graduation_rate",
        desc="University has a four-year graduation rate of at least 45%",
        parent=unode,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(uni.grad_rate and _nz(uni.grad_rate.four_year_rate).strip()),
        id=f"u{idx+1}_rate_identified",
        desc="Four-year graduation rate is identified",
        parent=grad_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(uni.grad_rate and len(_safe_sources(uni.grad_rate.sources)) > 0),
        id=f"u{idx+1}_reference_url_graduation",
        desc="Reference URL provided for graduation rate",
        parent=grad_node,
        critical=True
    )
    grad_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_rate_meets_minimum",
        desc="Four-year graduation rate is 45% or higher",
        parent=grad_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The four-year graduation rate for bachelor's degree students at {uni_name} is at least 45%.",
        node=grad_leaf,
        sources=_safe_sources(uni.grad_rate.sources if uni.grad_rate else []),
        additional_instruction="Only consider a 'four-year' graduation rate. Do not accept 'six-year' or general graduation rates unless the page clearly states the four-year value. Accept reasonable rounding (e.g., 44.6% does not meet 45%)."
    )

    # ------------------------- Honors Program ---------------------------------
    honors_node = evaluator.add_parallel(
        id=f"u{idx+1}_honors_program",
        desc="University offers an honors program with stated admission criteria for incoming freshmen",
        parent=unode,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(uni.honors and len(_safe_sources(uni.honors.sources)) > 0),
        id=f"u{idx+1}_reference_url_honors",
        desc="Reference URL provided for honors program and admission criteria",
        parent=honors_node,
        critical=True
    )
    honors_exists_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_honors_program_existence",
        desc="Honors program exists and accepts incoming freshmen",
        parent=honors_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} has an honors program (e.g., {_nz(uni.honors.program_name if uni.honors else '')}) that accepts incoming freshmen applicants.",
        node=honors_exists_leaf,
        sources=_safe_sources(uni.honors.sources if uni.honors else []),
        additional_instruction="Pass only if the page explicitly indicates the honors program exists and first-year/incoming freshmen can apply (or are eligible)."
    )

    honors_criteria_node = evaluator.add_parallel(
        id=f"u{idx+1}_honors_admission_criteria",
        desc="Honors program has publicly stated admission criteria including test scores and/or GPA",
        parent=honors_node,
        critical=True
    )
    criteria_metrics_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_criteria_includes_metrics",
        desc="Admission criteria includes at least one of: minimum SAT score, minimum ACT score, or minimum high school GPA",
        parent=honors_criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim="The honors program's published admission criteria include at least one explicit minimum threshold among: SAT score, ACT score, or high school GPA.",
        node=criteria_metrics_leaf,
        sources=_safe_sources(uni.honors.sources if uni.honors else []),
        additional_instruction="Look for explicit minimums like 'minimum ACT 28', 'SAT 1300+', or 'GPA 3.5 or higher'. Phrases like 'typical', 'average', or 'recommended' without a stated minimum do not satisfy this requirement."
    )
    criteria_public_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_criteria_publicly_available",
        desc="Admission criteria are publicly available on official university website",
        parent=honors_criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim="The honors admission criteria (including any minimum SAT/ACT/GPA thresholds) are publicly posted on an official university website.",
        node=criteria_public_leaf,
        sources=_safe_sources(uni.honors.sources if uni.honors else []),
        additional_instruction="Prefer '.edu' domains or official subdomains. The content must clearly be an official university page, not an unaffiliated third-party site."
    )

    # ------------------------- Study Abroad -----------------------------------
    abroad_node = evaluator.add_parallel(
        id=f"u{idx+1}_study_abroad",
        desc="University offers study abroad programs available to undergraduate students",
        parent=unode,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(uni.study_abroad and len(_safe_sources(uni.study_abroad.sources)) > 0),
        id=f"u{idx+1}_reference_url_study_abroad",
        desc="Reference URL provided for study abroad programs",
        parent=abroad_node,
        critical=True
    )
    abroad_exist_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_abroad_programs_exist",
        desc="Study abroad programs are offered",
        parent=abroad_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} offers study abroad programs.",
        node=abroad_exist_leaf,
        sources=_safe_sources(uni.study_abroad.sources if uni.study_abroad else []),
        additional_instruction="The page should indicate study abroad programs exist (provider lists, office pages, or program catalogs)."
    )
    abroad_undergrad_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_abroad_undergrad_eligible",
        desc="Programs are available to undergraduate students",
        parent=abroad_node,
        critical=True
    )
    await evaluator.verify(
        claim="These study abroad programs are available to undergraduate students.",
        node=abroad_undergrad_leaf,
        sources=_safe_sources(uni.study_abroad.sources if uni.study_abroad else []),
        additional_instruction="Look for explicit undergraduate eligibility or statements that undergraduate students can participate."
    )

    # ------------------------- Recreation/fitness ------------------------------
    rec_node = evaluator.add_parallel(
        id=f"u{idx+1}_recreation_facilities",
        desc="University has on-campus recreation or fitness facilities available to students",
        parent=unode,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(uni.recreation and len(_safe_sources(uni.recreation.sources)) > 0),
        id=f"u{idx+1}_reference_url_recreation",
        desc="Reference URL provided for recreation facilities",
        parent=rec_node,
        critical=True
    )
    rec_exist_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_facilities_exist",
        desc="On-campus recreation or fitness facilities are identified",
        parent=rec_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} has on-campus recreation or fitness facilities.",
        node=rec_exist_leaf,
        sources=_safe_sources(uni.recreation.sources if uni.recreation else []),
        additional_instruction="Look for campus recreation center pages, gym/fitness facility pages, with on-campus indication."
    )
    rec_access_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_student_access",
        desc="Facilities are available to students",
        parent=rec_node,
        critical=True
    )
    await evaluator.verify(
        claim="These recreation/fitness facilities are available to students.",
        node=rec_access_leaf,
        sources=_safe_sources(uni.recreation.sources if uni.recreation else []),
        additional_instruction="Confirm student eligibility or access (e.g., membership included in student fees or student membership available)."
    )

    # ------------------------- Career services --------------------------------
    career_node = evaluator.add_parallel(
        id=f"u{idx+1}_career_services",
        desc="University provides career services including internship or job placement support",
        parent=unode,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(uni.career and len(_safe_sources(uni.career.sources)) > 0),
        id=f"u{idx+1}_reference_url_career",
        desc="Reference URL provided for career services",
        parent=career_node,
        critical=True
    )
    career_exist_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_services_exist",
        desc="Career services office or program is identified",
        parent=career_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} provides career services (e.g., a career center or equivalent).",
        node=career_exist_leaf,
        sources=_safe_sources(uni.career.sources if uni.career else []),
        additional_instruction="Look for 'Career Center', 'Career Services', or similar official pages."
    )
    career_place_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_placement_support",
        desc="Services include internship or job placement support",
        parent=career_node,
        critical=True
    )
    await evaluator.verify(
        claim="Career services include internship or job placement support for students.",
        node=career_place_leaf,
        sources=_safe_sources(uni.career.sources if uni.career else []),
        additional_instruction="Accept offerings such as internship advising, job placement assistance, on-campus recruiting, career fairs with employer connections, or similar."
    )

    # ------------------------- Dining facilities ------------------------------
    dining_node = evaluator.add_parallel(
        id=f"u{idx+1}_dining_facilities",
        desc="University has on-campus dining facilities with meal plan options",
        parent=unode,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(uni.dining and len(_safe_sources(uni.dining.sources)) > 0),
        id=f"u{idx+1}_reference_url_dining",
        desc="Reference URL provided for dining and meal plans",
        parent=dining_node,
        critical=True
    )
    dining_exist_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_dining_exists",
        desc="On-campus dining facilities are identified",
        parent=dining_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} has on-campus dining facilities.",
        node=dining_exist_leaf,
        sources=_safe_sources(uni.dining.sources if uni.dining else []),
        additional_instruction="Look for official dining services pages describing dining halls, food courts, etc., located on campus."
    )
    dining_meal_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_meal_plans_available",
        desc="Meal plan options are available to students",
        parent=dining_node,
        critical=True
    )
    await evaluator.verify(
        claim="Meal plan options are available to students.",
        node=dining_meal_leaf,
        sources=_safe_sources(uni.dining.sources if uni.dining else []),
        additional_instruction="The page should explicitly mention student meal plans, dining plans, or equivalent subscription options."
    )

    # ------------------------- Transportation ---------------------------------
    trans_node = evaluator.add_parallel(
        id=f"u{idx+1}_transportation",
        desc="University provides campus shuttle or transportation services",
        parent=unode,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(uni.transport and len(_safe_sources(uni.transport.sources)) > 0),
        id=f"u{idx+1}_reference_url_transportation",
        desc="Reference URL provided for transportation services",
        parent=trans_node,
        critical=True
    )
    trans_exist_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_transport_services_exist",
        desc="Campus shuttle or transportation services are identified",
        parent=trans_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} provides campus shuttle or transportation services.",
        node=trans_exist_leaf,
        sources=_safe_sources(uni.transport.sources if uni.transport else []),
        additional_instruction="Look for 'shuttle', 'campus bus', 'transit', or equivalent services operated or provided for the campus."
    )
    trans_access_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_student_access_transport",
        desc="Services are available to students",
        parent=trans_node,
        critical=True
    )
    await evaluator.verify(
        claim="These transportation services are available to students.",
        node=trans_access_leaf,
        sources=_safe_sources(uni.transport.sources if uni.transport else []),
        additional_instruction="The page should indicate students are eligible to use the shuttle/transit services."
    )

    # ------------------------- Campus safety ----------------------------------
    safety_node = evaluator.add_parallel(
        id=f"u{idx+1}_campus_safety",
        desc="University has campus safety or security services operating 24/7",
        parent=unode,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(uni.safety and len(_safe_sources(uni.safety.sources)) > 0),
        id=f"u{idx+1}_reference_url_safety",
        desc="Reference URL provided for campus safety services",
        parent=safety_node,
        critical=True
    )
    safety_exist_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_safety_services_exist",
        desc="Campus safety or security services are identified",
        parent=safety_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} has campus safety or campus police/security services.",
        node=safety_exist_leaf,
        sources=_safe_sources(uni.safety.sources if uni.safety else []),
        additional_instruction="Look for 'campus police', 'public safety', 'security', or equivalent unit."
    )
    safety_247_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_operates_24_7",
        desc="Services operate 24 hours a day, 7 days a week",
        parent=safety_node,
        critical=True
    )
    await evaluator.verify(
        claim="Campus safety/security services operate 24 hours a day, 7 days a week.",
        node=safety_247_leaf,
        sources=_safe_sources(uni.safety.sources if uni.safety else []),
        additional_instruction="Look specifically for '24/7', '24 hours', 'around-the-clock', or equivalent phrasing."
    )

    # ------------------------- Public institution & US location ---------------
    public_node = evaluator.add_parallel(
        id=f"u{idx+1}_public_university",
        desc="University is a public institution in the United States",
        parent=unode,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(uni.institution and len(_safe_sources(uni.institution.sources)) > 0),
        id=f"u{idx+1}_reference_url_institution",
        desc="Reference URL provided confirming institutional status",
        parent=public_node,
        critical=True
    )
    public_status_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_public_status",
        desc="University is identified as a public institution",
        parent=public_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} is a public university.",
        node=public_status_leaf,
        sources=_safe_sources(uni.institution.sources if uni.institution else []),
        additional_instruction="Accept phrases like 'public university', 'public research university', 'state university', or official classification indicating public control."
    )
    # Location / US
    loc_part = f" in {city}, {state}" if city and state else (f" in {state}" if state else "")
    us_location_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_us_location",
        desc="University is located in the United States",
        parent=public_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The university is located in the United States{loc_part}.",
        node=us_location_leaf,
        sources=_safe_sources(uni.institution.sources if uni.institution else []),
        additional_instruction="The page should clearly indicate a U.S. location. If city/state are provided, they should match; otherwise, confirming U.S. location suffices."
    )


# -----------------------------------------------------------------------------
# Main evaluation entry
# -----------------------------------------------------------------------------
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Normalize to exactly three entries (pad with empty objects if fewer)
    universities: List[University] = list(extracted.universities or [])
    while len(universities) < 3:
        universities.append(University())
    universities = universities[:3]

    # Build university verification subtrees
    tasks = []
    for i in range(3):
        tasks.append(verify_university(evaluator, root, universities[i], i))
    await asyncio.gather(*tasks)

    return evaluator.get_summary()