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
TASK_ID = "nc_sc_grad_reqs_2026_27"
TASK_DESCRIPTION = """
Create a comprehensive graduation requirements comparison for public high schools in North Carolina and South Carolina for students entering 9th grade in the 2026-27 school year. For each state, document: (1) Total credit requirement for graduation; (2) Required credits by subject area including English (number of credits and specific course sequence), Mathematics (number of credits and specific course requirements), Science (number of credits and specific course requirements), Social Studies (number of credits and specific course requirements), Health/Physical Education (number of credits and any special mandates), Elective credits (total count and any category-specific requirements), and Computer Science requirement; (3) State-specific additional requirements for the 2026-27 cohort; (4) Minimum instructional time requirements (days and/or hours); (5) For North Carolina: arts education requirement for applicable student cohorts. Provide specific credit counts, required course names where mandated by state policy, and authoritative source URLs from state education departments.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class SubjectRequirement(BaseModel):
    credits: Optional[str] = None
    required_courses: List[str] = Field(default_factory=list)
    special_mandates: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ElectivesRequirement(BaseModel):
    total_elective_credits: Optional[str] = None
    category_constraints: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class InstructionalTime(BaseModel):
    min_days_or_hours_statement: Optional[str] = None
    min_days: Optional[str] = None
    min_hours: Optional[str] = None
    calendar_length_requirement: Optional[str] = None
    vacation_days_requirement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ArtsEducationRequirement(BaseModel):
    statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class StateRequirements(BaseModel):
    state_name: Optional[str] = None
    total_credits: Optional[str] = None

    english: SubjectRequirement = Field(default_factory=SubjectRequirement)
    math: SubjectRequirement = Field(default_factory=SubjectRequirement)
    science: SubjectRequirement = Field(default_factory=SubjectRequirement)
    social_studies: SubjectRequirement = Field(default_factory=SubjectRequirement)
    health_pe: SubjectRequirement = Field(default_factory=SubjectRequirement)
    computer_science: SubjectRequirement = Field(default_factory=SubjectRequirement)

    electives: ElectivesRequirement = Field(default_factory=ElectivesRequirement)

    additional_requirements_2026_27: Optional[str] = None

    arts_education: ArtsEducationRequirement = Field(default_factory=ArtsEducationRequirement)
    instructional_time: InstructionalTime = Field(default_factory=InstructionalTime)

    state_sources: List[str] = Field(default_factory=list)


class ComparisonExtraction(BaseModel):
    north_carolina: StateRequirements = Field(default_factory=StateRequirements)
    south_carolina: StateRequirements = Field(default_factory=StateRequirements)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_comparison() -> str:
    return """
    Extract a structured comparison of high school graduation requirements as presented in the answer for North Carolina and South Carolina, specifically for students entering grade 9 in the 2026–27 school year.

    For each state, extract the following fields:

    - state_name: The state's full name (e.g., "North Carolina", "South Carolina").
    - total_credits: The total number of credits required to graduate as explicitly stated.
    
    - english:
        - credits: Number of English credits required.
        - required_courses: List each specifically-mandated English course or sequence (e.g., ["English I", "English II", "English III", "English IV"]) if stated.
        - special_mandates: Any special mandates linked to English, if mentioned.
        - sources: URL(s) cited in the answer that support the English requirement.
    
    - math:
        - credits
        - required_courses: List mandated courses (e.g., ["NC Math 1", "NC Math 2", "NC Math 3", "4th math aligned to postsecondary plans"]) if stated.
        - special_mandates
        - sources
    
    - science:
        - credits
        - required_courses: List mandated courses (e.g., ["Physical science", "Biology", "Earth/Environmental science"]) if stated.
        - special_mandates
        - sources
    
    - social_studies:
        - credits
        - required_courses: List mandated courses (e.g., for NC: ["Founding Principles/Civic Literacy", "Economics and Personal Finance", "American History", "World History"]; for SC include breakdown such as US History, Government, Economics, Other SS as stated).
        - special_mandates
        - sources
    
    - health_pe:
        - credits
        - special_mandates: Include CPR or other mandates or allowed alternatives/substitutions if explicitly stated in the answer.
        - sources
    
    - computer_science:
        - credits: If a Computer Science credit is required (e.g., NC 2026–27: "1"), provide it.
        - required_courses: If a specific named CS course is mandated, list it; else leave empty.
        - special_mandates
        - sources
    
    - electives:
        - total_elective_credits
        - category_constraints: Text describing category-specific elective requirements (e.g., NC: "2 credits from any combination of CTE, Arts Education, or World Language...").
        - sources
    
    - additional_requirements_2026_27: Text summarizing any additional cohort-specific state requirements for the 2026–27 entering class beyond those above. If the answer explicitly states none, put "none" (lowercase).
    
    - arts_education (North Carolina-specific; if stated):
        - statement: The arts education requirement wording for applicable cohorts (e.g., "Students entering Grade 6 in fall 2022 or later must complete at least one arts education course in grades 6–12...").
        - sources
    
    - instructional_time:
        - min_days_or_hours_statement: The exact phrasing provided in the answer regarding the minimum instructional days and/or hours.
        - min_days: If a specific minimum days figure is stated, include it; else null.
        - min_hours: If a specific minimum hours figure is stated, include it; else null.
        - calendar_length_requirement: If the answer states a required calendar coverage (e.g., "at least nine calendar months"), include it.
        - vacation_days_requirement: If the answer states a minimum vacation days requirement (e.g., "minimum 10 annual vacation days"), include it.
        - sources
    
    - state_sources: All authoritative state education department or state statute/regulation URLs the answer cites for this state (e.g., dpi.nc.gov, ncleg.gov, ed.sc.gov). Include only URLs explicitly present in the answer.

    IMPORTANT:
    - Extract only what is explicitly present in the answer text.
    - Return null for any missing scalar field; return empty lists for missing list fields.
    - Always extract URLs exactly as they appear and ensure they are valid.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def collect_state_urls(state: StateRequirements) -> List[str]:
    urls: List[str] = []
    urls.extend(state.english.sources or [])
    urls.extend(state.math.sources or [])
    urls.extend(state.science.sources or [])
    urls.extend(state.social_studies.sources or [])
    urls.extend(state.health_pe.sources or [])
    urls.extend(state.computer_science.sources or [])
    urls.extend(state.electives.sources or [])
    urls.extend(state.instructional_time.sources or [])
    urls.extend(state.arts_education.sources or [])
    urls.extend(state.state_sources or [])
    return _dedup_urls(urls)


def choose_sources(area_sources: Optional[List[str]], fallback_state_urls: List[str]) -> List[str]:
    if area_sources and len(area_sources) > 0:
        return _dedup_urls(area_sources)
    return _dedup_urls(fallback_state_urls)


async def verify_with_urls_or_require_citation(
    evaluator: Evaluator,
    claim: str,
    node,
    urls: List[str],
    add_ins: str,
) -> None:
    """
    Verify a claim using multi-URL evidence when available; if no URLs are present,
    fall back to simple verification but explicitly require that the answer cites
    authoritative URLs for this point (so the judge should fail if the answer omits citations).
    """
    if urls:
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction=add_ins,
        )
    else:
        # No URLs present – require the judge to mark incorrect if citations are missing.
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=None,
            additional_instruction=(
                add_ins
                + "\nCRITICAL: Only mark Correct if the answer explicitly cites authoritative state URL(s) for this point. "
                  "If the answer provides no URLs for this point, mark Incorrect."
            ),
        )


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_nc_subtree(evaluator: Evaluator, parent_node, nc: StateRequirements) -> None:
    # Parent node for North Carolina (critical)
    nc_parent = evaluator.add_parallel(
        id="North_Carolina",
        desc="North Carolina graduation requirements for students entering 9th grade in 2026–27 (Future-Ready Course of Study), with required details and sources.",
        parent=parent_node,
        critical=True,
    )

    state_urls = collect_state_urls(nc)

    # NC_Total_Credits
    leaf = evaluator.add_leaf(
        id="NC_Total_Credits",
        desc="States the total credit requirement for graduation (must match constraints: at least 22 credits).",
        parent=nc_parent,
        critical=True,
    )
    total_claim_val = nc.total_credits if (nc and nc.total_credits) else "at least 22"
    claim = (
        f"For North Carolina (students entering grade 9 in 2026–27), the answer explicitly states a total graduation "
        f"requirement of {total_claim_val} credits, and this is at least 22 credits as supported by the cited state source(s)."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources([], state_urls),
        add_ins="Confirm both that the answer states a specific total credit count and that the official NC source supports it; also ensure the total is ≥ 22.",
    )

    # NC_English_Requirement
    leaf = evaluator.add_leaf(
        id="NC_English_Requirement",
        desc="English requirement includes required credits and mandated course sequence (must match constraints: 4 sequential credits: English I, English II, English III, English IV).",
        parent=nc_parent,
        critical=True,
    )
    claim = (
        "For North Carolina (2026–27 cohort), the answer states that English requires 4 credits and the specific course "
        "sequence English I, English II, English III, and English IV; this matches the official NC policy."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources(nc.english.sources, state_urls),
        add_ins="Only pass if the answer explicitly lists the sequence English I–IV and the cited NC source confirms it.",
    )

    # NC_Math_Requirement
    leaf = evaluator.add_leaf(
        id="NC_Math_Requirement",
        desc="Mathematics requirement includes required credits and mandated course requirements (must match constraints: 4 credits including NC Math 1, NC Math 2, NC Math 3, and a 4th math course aligned with post-high school plans).",
        parent=nc_parent,
        critical=True,
    )
    claim = (
        "For North Carolina (2026–27 cohort), the answer states Mathematics requires 4 credits including NC Math 1, NC Math 2, "
        "NC Math 3, and a fourth math course aligned with the student's post–high school plans; this matches the official NC policy."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources(nc.math.sources, state_urls),
        add_ins="Verify the exact math course names and the flexible 4th math requirement are stated in the answer and supported by NC sources.",
    )

    # NC_Science_Requirement
    leaf = evaluator.add_leaf(
        id="NC_Science_Requirement",
        desc="Science requirement includes required credits and mandated course requirements (must match constraints: 3 credits including a physical science course, Biology, and an earth/environmental science course).",
        parent=nc_parent,
        critical=True,
    )
    claim = (
        "For North Carolina (2026–27 cohort), the answer states Science requires 3 credits including a physical science course, "
        "Biology, and an earth/environmental science course; this matches the official NC policy."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources(nc.science.sources, state_urls),
        add_ins="Confirm that all three science components appear in the answer and are supported by the NC source(s).",
    )

    # NC_Social_Studies_Requirement
    leaf = evaluator.add_leaf(
        id="NC_Social_Studies_Requirement",
        desc="Social Studies requirement includes required credits and mandated course requirements (must match constraints: 4 credits including Founding Principles/Civic Literacy, Economics and Personal Finance, American History, World History).",
        parent=nc_parent,
        critical=True,
    )
    claim = (
        "For North Carolina (2026–27 cohort), the answer states Social Studies requires 4 credits including "
        "Founding Principles/Civic Literacy, Economics and Personal Finance, American History, and World History; "
        "this matches the official NC policy."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources(nc.social_studies.sources, state_urls),
        add_ins="Ensure the answer names all four NC-required Social Studies courses and that the NC source(s) confirm them.",
    )

    # NC_Health_PE_Requirement
    leaf = evaluator.add_leaf(
        id="NC_Health_PE_Requirement",
        desc="Health/Physical Education requirement includes required credit amount and special mandate (must match constraints: 1 credit including successful completion of CPR instruction).",
        parent=nc_parent,
        critical=True,
    )
    claim = (
        "For North Carolina (2026–27 cohort), the answer states Health/Physical Education requires 1 credit and includes "
        "successful completion of CPR instruction; this matches the official NC policy."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources(nc.health_pe.sources, state_urls),
        add_ins="Accept only if the answer explicitly mentions 1 credit and the CPR instruction mandate, confirmed by NC sources.",
    )

    # NC_Computer_Science_Requirement_2026_27
    leaf = evaluator.add_leaf(
        id="NC_Computer_Science_Requirement_2026_27",
        desc="Includes the cohort-specific Computer Science credit requirement (must match constraints: students entering grade 9 in 2026–27 must take 1 Computer Science credit).",
        parent=nc_parent,
        critical=True,
    )
    claim = (
        "For North Carolina, students entering grade 9 in 2026–27 must complete 1 Computer Science credit; the answer "
        "states this and it is supported by official NC sources."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources(nc.computer_science.sources, state_urls),
        add_ins="Pass only if the answer clearly indicates a 1-credit Computer Science requirement beginning with the 2026–27 9th-grade cohort, supported by NC sources.",
    )

    # NC_Electives_Requirement_2026_27 (parallel group)
    electives_group = evaluator.add_parallel(
        id="NC_Electives_Requirement_2026_27",
        desc="States NC elective credit requirements applicable to the 2026–27 entering-9th-grade cohort, incorporating the constraint that elective credits are reduced from 6 to 5 due to the new Computer Science requirement.",
        parent=nc_parent,
        critical=True,
    )

    # NC_Electives_Total_2026_27
    leaf = evaluator.add_leaf(
        id="NC_Electives_Total_2026_27",
        desc="States the total elective credits required for the 2026–27 cohort (must match constraints: additional elective credits reduced from 6 to 5).",
        parent=electives_group,
        critical=True,
    )
    claim = (
        "For North Carolina (2026–27 cohort), the answer states that the additional elective credits are reduced from 6 to 5 "
        "because of the new Computer Science credit; this statement is supported by official NC sources."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources(nc.electives.sources, state_urls),
        add_ins="Ensure the answer explicitly notes the reduction from 6 to 5 elective credits for the 2026–27 cohort and cites NC sources.",
    )

    # NC_Electives_Category_Constraints
    leaf = evaluator.add_leaf(
        id="NC_Electives_Category_Constraints",
        desc="States the elective category-specific requirements (must match constraints: 2 elective credits from any combination of CTE, Arts Education, or World Language; remaining elective credits drawn from allowed areas such as CTE/ROTC/Arts/other subject areas/cross-disciplinary courses).",
        parent=electives_group,
        critical=True,
    )
    claim = (
        "For North Carolina, the answer states that 2 elective credits must come from any combination of CTE, Arts Education, "
        "or World Language, and the remaining elective credits may be from allowed areas such as CTE/ROTC/Arts/other subject "
        "areas/cross-disciplinary courses; this matches NC policy per sources."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources(nc.electives.sources, state_urls),
        add_ins="Pass only if the answer includes both the 2-credit category rule and the permitted areas for remaining electives, supported by NC sources.",
    )

    # NC_Arts_Education_Requirement
    leaf = evaluator.add_leaf(
        id="NC_Arts_Education_Requirement",
        desc="Includes the NC arts education requirement for applicable cohorts (must match constraints: students entering Grade 6 in fall 2022 or later must complete at least one arts education course in grades 6–12, completing the standard course of study in its entirety).",
        parent=nc_parent,
        critical=True,
    )
    claim = (
        "The answer states North Carolina's arts education requirement for applicable cohorts: students entering Grade 6 in fall 2022 "
        "or later must complete at least one arts education course in grades 6–12, completing the standard course of study in its entirety; "
        "this is supported by official NC sources."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources(nc.arts_education.sources, state_urls),
        add_ins="Ensure the answer states the cohort applicability (entering Grade 6 in fall 2022 or later) and the 'one arts course in grades 6–12' requirement, with NC citation.",
    )

    # NC_Other_Cohort_Specific_Additional_Requirements_2026_27
    leaf = evaluator.add_leaf(
        id="NC_Other_Cohort_Specific_Additional_Requirements_2026_27",
        desc="Identifies any other NC state-specific additional requirements for the 2026–27 entering-9th-grade cohort beyond the requirements already listed above, OR explicitly states that none are identified.",
        parent=nc_parent,
        critical=True,
    )
    addl_text = nc.additional_requirements_2026_27 or "none"
    claim = (
        "Regarding other North Carolina cohort-specific graduation requirements for students entering 9th grade in 2026–27, "
        f"the answer states: '{addl_text}'. This is accurate according to the cited official NC sources (i.e., no other additional "
        "cohort-specific requirements are imposed unless explicitly indicated by sources)."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources([], state_urls),
        add_ins="Pass if either (a) the answer lists additional cohort-specific requirements and the NC sources confirm them, or (b) the answer explicitly states none and sources do not indicate additional cohort-specific requirements.",
    )

    # NC_Instructional_Time group
    inst_group = evaluator.add_parallel(
        id="NC_Instructional_Time",
        desc="States NC minimum instructional time requirements (must match constraints).",
        parent=nc_parent,
        critical=True,
    )

    # NC_Min_Days_or_Hours
    leaf = evaluator.add_leaf(
        id="NC_Min_Days_or_Hours",
        desc="States NC minimum instructional days or hours (must match constraints: minimum 185 days or 1,025 hours).",
        parent=inst_group,
        critical=True,
    )
    claim = (
        "North Carolina requires a minimum of 185 instructional days or 1,025 instructional hours; the answer explicitly states this, "
        "and it is supported by official NC sources."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources(nc.instructional_time.sources, state_urls),
        add_ins="Ensure the answer includes both the 185-day and 1,025-hour options and that the NC source confirms these minimums.",
    )

    # NC_Min_Calendar_Length
    leaf = evaluator.add_leaf(
        id="NC_Min_Calendar_Length",
        desc="States NC minimum calendar coverage requirement (must match constraints: at least nine calendar months).",
        parent=inst_group,
        critical=True,
    )
    claim = (
        "North Carolina requires a school calendar covering at least nine calendar months; the answer states this and it is supported by official NC sources."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources(nc.instructional_time.sources, state_urls),
        add_ins="Confirm the answer explicitly mentions 'at least nine calendar months' and that the NC source verifies it.",
    )

    # NC_Min_Vacation_Days
    leaf = evaluator.add_leaf(
        id="NC_Min_Vacation_Days",
        desc="States NC minimum annual vacation days requirement (must match constraints: minimum 10 annual vacation days).",
        parent=inst_group,
        critical=True,
    )
    claim = (
        "North Carolina requires a minimum of 10 annual vacation days; the answer includes this requirement and it is supported by official NC sources."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources(nc.instructional_time.sources, state_urls),
        add_ins="Ensure the answer states 'minimum 10 annual vacation days' and the NC source confirms it.",
    )

    # NC_Sources
    leaf = evaluator.add_leaf(
        id="NC_Sources",
        desc="Provides authoritative NC state education authority URL(s) that support the NC requirements stated.",
        parent=nc_parent,
        critical=True,
    )
    claim = (
        "At least one cited North Carolina source is an authoritative state page (e.g., NC Department of Public Instruction dpi.nc.gov or ncpublicschools.gov, or an official ncleg.gov statute/regulation) and it directly pertains to graduation requirements/policies for the relevant cohort."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        state_urls,
        add_ins="Fail if no NC URLs are provided. Pass if at least one .gov state education authority page relevant to graduation requirements is cited.",
    )


async def build_sc_subtree(evaluator: Evaluator, parent_node, sc: StateRequirements) -> None:
    # Parent node for South Carolina (critical)
    sc_parent = evaluator.add_parallel(
        id="South_Carolina",
        desc="South Carolina graduation requirements for students entering 9th grade in 2026–27, including required details and sources.",
        parent=parent_node,
        critical=True,
    )

    state_urls = collect_state_urls(sc)

    # SC_Total_Credits
    leaf = evaluator.add_leaf(
        id="SC_Total_Credits",
        desc="States the total credit requirement for graduation (must match constraints: 24 total credits).",
        parent=sc_parent,
        critical=True,
    )
    claim = (
        "For South Carolina, the answer explicitly states a total graduation requirement of 24 credits, supported by official SC sources."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources([], state_urls),
        add_ins="Only pass if the answer explicitly states '24 credits' and the cited SC source confirms this.",
    )

    # SC_English_Requirement
    leaf = evaluator.add_leaf(
        id="SC_English_Requirement",
        desc="States required English credits (must match constraints: 4) and includes any state-mandated course sequence/names if specified by state policy (otherwise indicates only credit count is specified).",
        parent=sc_parent,
        critical=True,
    )
    claim = (
        "For South Carolina, the answer states English requires 4 credits. If state policy mandates a specific sequence/named courses, the answer includes those names; "
        "otherwise, only the 4-credit requirement is claimed. This matches the official SC source(s)."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources(sc.english.sources, state_urls),
        add_ins="Confirm the answer explicitly states '4 English credits'. If specific course names are mandated in the SC source, ensure the answer lists them; if not mandated, stating only the credit count is acceptable.",
    )

    # SC_Math_Requirement
    leaf = evaluator.add_leaf(
        id="SC_Math_Requirement",
        desc="States required Mathematics credits (must match constraints: 4) and includes any state-mandated course requirements/names if specified by state policy (otherwise indicates only credit count is specified).",
        parent=sc_parent,
        critical=True,
    )
    claim = (
        "For South Carolina, the answer states Mathematics requires 4 credits; if the state mandates specific course names, the answer includes them; "
        "if not, only the credit count is stated. This is supported by SC sources."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources(sc.math.sources, state_urls),
        add_ins="Ensure the answer clearly states '4 Math credits' and, if SC policy specifies named courses, that those names appear and match the sources.",
    )

    # SC_Science_Requirement
    leaf = evaluator.add_leaf(
        id="SC_Science_Requirement",
        desc="States required Science credits (must match constraints: 3) and includes any state-mandated course requirements/names if specified by state policy (otherwise indicates only credit count is specified).",
        parent=sc_parent,
        critical=True,
    )
    claim = (
        "For South Carolina, the answer states Science requires 3 credits; if specific course names are mandated by the state, the answer includes them; "
        "otherwise only the credit count is stated. This matches the official SC sources."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources(sc.science.sources, state_urls),
        add_ins="Verify the answer explicitly includes '3 Science credits'. If SC sources specify named courses, ensure the answer names them accordingly.",
    )

    # SC_Social_Studies_Requirement
    leaf = evaluator.add_leaf(
        id="SC_Social_Studies_Requirement",
        desc="States required Social Studies credits and breakdown (must match constraints: 1 U.S. History credit, 0.5 Economics credit, 0.5 Government credit, and 1 Other Social Studies credit).",
        parent=sc_parent,
        critical=True,
    )
    claim = (
        "For South Carolina, the answer states the Social Studies breakdown: 1 credit U.S. History, 0.5 Economics, 0.5 Government, and 1 Other Social Studies credit; "
        "this matches the official SC policy per sources."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources(sc.social_studies.sources, state_urls),
        add_ins="Ensure the answer includes the exact SS breakdown and that it is confirmed by SC sources.",
    )

    # SC_Health_PE_Requirement
    leaf = evaluator.add_leaf(
        id="SC_Health_PE_Requirement",
        desc="States Physical Education requirement including allowed alternatives (must match constraints: 1 PE credit with allowed alternative noted) and states whether any separate Health credit is required or not required per SC policy.",
        parent=sc_parent,
        critical=True,
    )
    claim = (
        "For South Carolina, the answer states a 1-credit Physical Education requirement and mentions allowed alternatives/substitutions as specified by state policy, and it states whether a separate Health credit is required or not; this aligns with official SC sources."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources(sc.health_pe.sources, state_urls),
        add_ins="Pass only if the answer explicitly mentions '1 PE credit', includes allowed alternatives if policy specifies them, and clarifies the Health credit status, supported by SC sources.",
    )

    # SC_Computer_Science_Requirement
    leaf = evaluator.add_leaf(
        id="SC_Computer_Science_Requirement",
        desc="States Computer Science credit requirement (must match constraints: 1 Computer Science credit).",
        parent=sc_parent,
        critical=True,
    )
    claim = (
        "For South Carolina, the answer states a 1-credit Computer Science graduation requirement, supported by official SC sources."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources(sc.computer_science.sources, state_urls),
        add_ins="Ensure the answer explicitly states '1 Computer Science credit' and the SC source confirms it.",
    )

    # SC_Personal_Finance_Requirement
    leaf = evaluator.add_leaf(
        id="SC_Personal_Finance_Requirement",
        desc="States Personal Finance requirement (must match constraints: 0.5 credit).",
        parent=sc_parent,
        critical=True,
    )
    claim = (
        "For South Carolina, the answer states a 0.5-credit Personal Finance requirement, supported by official SC sources."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources(sc.social_studies.sources + sc.state_sources, state_urls),
        add_ins="Confirm the answer clearly includes '0.5 credit Personal Finance' and that SC sources corroborate it.",
    )

    # SC_Electives_Requirement
    leaf = evaluator.add_leaf(
        id="SC_Electives_Requirement",
        desc="States elective credits required and elective category-specific constraints (must match constraints: 6.5 elective credits total AND 1 World Language or Career and Technology Elective credit requirement).",
        parent=sc_parent,
        critical=True,
    )
    claim = (
        "For South Carolina, the answer states 6.5 elective credits total and an additional requirement of 1 credit in World Language or Career and Technology Education (CTE) elective; this matches SC policy per sources."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources(sc.electives.sources, state_urls),
        add_ins="Ensure both the '6.5 electives' and '1 WL or CTE elective' requirements are explicitly included in the answer and confirmed by SC sources.",
    )

    # SC_Instructional_Time
    leaf = evaluator.add_leaf(
        id="SC_Instructional_Time",
        desc="States SC minimum instructional time requirements (days and/or hours) for the relevant cohort (no hard-coded numeric value unless explicitly provided in constraints/question).",
        parent=sc_parent,
        critical=True,
    )
    stmt = sc.instructional_time.min_days_or_hours_statement or "the state's minimum instructional time requirement (days and/or hours)"
    claim = (
        f"For South Carolina, the answer accurately states {stmt} and cites official SC sources; "
        "the statement matches what is shown on the cited source(s)."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources(sc.instructional_time.sources, state_urls),
        add_ins="Pass only if the answer's statement about minimum instructional time aligns with the cited official SC source(s). Do not require a specific numeric value beyond what the answer and sources state.",
    )

    # SC_Other_Cohort_Specific_Additional_Requirements_2026_27
    leaf = evaluator.add_leaf(
        id="SC_Other_Cohort_Specific_Additional_Requirements_2026_27",
        desc="Identifies any SC state-specific additional requirements for the 2026–27 entering-9th-grade cohort beyond the requirements already listed above, OR explicitly states that none are identified.",
        parent=sc_parent,
        critical=True,
    )
    addl_text = sc.additional_requirements_2026_27 or "none"
    claim = (
        "Regarding other South Carolina cohort-specific graduation requirements for students entering 9th grade in 2026–27, "
        f"the answer states: '{addl_text}'. This is accurate according to the cited official SC sources (i.e., no other additional "
        "cohort-specific requirements are imposed unless explicitly indicated by sources)."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        choose_sources([], state_urls),
        add_ins="Pass if either (a) the answer lists additional cohort-specific requirements and the SC sources confirm them, or (b) the answer explicitly states none and sources do not indicate additional cohort-specific requirements.",
    )

    # SC_Sources
    leaf = evaluator.add_leaf(
        id="SC_Sources",
        desc="Provides authoritative SC state education authority URL(s) that support the SC requirements stated.",
        parent=sc_parent,
        critical=True,
    )
    claim = (
        "At least one cited South Carolina source is an authoritative state page (e.g., SC Department of Education ed.sc.gov or other official sc.gov state pages) and it directly pertains to graduation requirements/policies for the relevant cohort."
    )
    await verify_with_urls_or_require_citation(
        evaluator,
        claim,
        leaf,
        state_urls,
        add_ins="Fail if no SC URLs are provided. Pass if at least one .gov state education authority page relevant to graduation requirements is cited.",
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
    Evaluate an answer for the NC/SC graduation requirements (2026–27 cohort) task.
    """
    # Initialize evaluator (root is non-critical by framework default)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation parallel
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

    # Create a critical wrapper node to mirror rubric root criticality
    rubric_root = evaluator.add_parallel(
        id="Root",
        desc="Graduation requirements comparison for public high schools in North Carolina and South Carolina for students entering 9th grade in 2026–27, including credit requirements by subject, cohort-specific additional requirements, instructional time, and authoritative state education department URLs.",
        parent=root,
        critical=True,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_comparison(),
        template_class=ComparisonExtraction,
        extraction_name="comparison_extraction",
    )

    # Build the North Carolina subtree
    await build_nc_subtree(evaluator, rubric_root, extracted.north_carolina or StateRequirements())

    # Build the South Carolina subtree
    await build_sc_subtree(evaluator, rubric_root, extracted.south_carolina or StateRequirements())

    # Return the final structured summary
    return evaluator.get_summary()