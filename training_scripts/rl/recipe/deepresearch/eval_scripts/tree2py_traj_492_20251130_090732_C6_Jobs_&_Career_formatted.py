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
TASK_ID = "career_transition_ivyleague_oc"
TASK_DESCRIPTION = """
A high school head football coach in Georgia is exploring career advancement opportunities in college coaching. The coach has 6 years of head coaching experience at a top Georgia high school program and holds a Master's degree in Education. They are specifically interested in offensive coordinator positions at Ivy League institutions and want to focus their research on Yale University and Harvard University.

For this career planning analysis, provide the following information:

1. Current Offensive Coordinators: For both Yale and Harvard, identify the current offensive coordinator by name, verify their official title, and determine how long they have served in their current role (measured in seasons as of the 2025 season).

2. Standard Qualification Requirements: Research and report the typical minimum qualification requirements for offensive coordinator positions at Division I FCS or Ivy League football programs, specifically: (a) the minimum educational degree required, and (b) the minimum number of years of coaching experience typically required.

3. Candidate Qualification Assessment: Based on the standard requirements you identified, assess whether the hypothetical candidate (Master's degree, 6 years head coaching experience) meets the minimum qualifications for offensive coordinator positions at these schools.

4. Financial Implications: Report the typical salary range for top high school head football coaches in Georgia and the approximate salary range for Ivy League offensive coordinator or assistant coach positions. Provide a brief comparison noting whether the transition would likely result in an increase or decrease in compensation.

For all information provided, include reference URLs from official athletic department websites, job postings, salary databases, or credible sports journalism sources to support your findings.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OCInfo(BaseModel):
    name: Optional[str] = None
    official_title: Optional[str] = None
    tenure_seasons_2025: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class QualificationRequirements(BaseModel):
    minimum_degree: Optional[str] = None
    degree_sources: List[str] = Field(default_factory=list)
    minimum_experience_years: Optional[str] = None
    experience_sources: List[str] = Field(default_factory=list)


class CandidateAssessment(BaseModel):
    education_meets_minimum: Optional[str] = None  # e.g., "meets", "does_not_meet", "unclear"
    experience_meets_minimum: Optional[str] = None
    overall_meets_minimum: Optional[str] = None


class FinancialImplications(BaseModel):
    georgia_hs_head_coach_salary_range: Optional[str] = None
    georgia_salary_sources: List[str] = Field(default_factory=list)
    ivy_league_oc_or_assistant_salary_range: Optional[str] = None
    ivy_salary_sources: List[str] = Field(default_factory=list)
    compensation_change: Optional[str] = None  # e.g., "increase", "decrease", "depends"


class CareerPlanningExtraction(BaseModel):
    yale: Optional[OCInfo] = None
    harvard: Optional[OCInfo] = None
    requirements: Optional[QualificationRequirements] = None
    assessment: Optional[CandidateAssessment] = None
    financials: Optional[FinancialImplications] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_career_planning() -> str:
    return """
    Extract the structured information required for the Ivy League offensive coordinator career planning analysis focusing on Yale and Harvard. Return a single JSON object strictly following the schema below:

    {
      "yale": {
        "name": string | null,
        "official_title": string | null,
        "tenure_seasons_2025": string | null,
        "sources": string[]  // URLs that support Yale OC identity/title/tenure
      },
      "harvard": {
        "name": string | null,
        "official_title": string | null,
        "tenure_seasons_2025": string | null,
        "sources": string[]  // URLs that support Harvard OC identity/title/tenure
      },
      "requirements": {
        "minimum_degree": string | null,                 // Typical minimum educational degree for OC roles at Div. I FCS / Ivy League
        "degree_sources": string[],                      // URLs supporting minimum_degree
        "minimum_experience_years": string | null,       // Typical minimum years of (coaching) experience for such roles
        "experience_sources": string[]                   // URLs supporting minimum_experience_years
      },
      "assessment": {
        "education_meets_minimum": string | null,        // "meets" / "does_not_meet" / "unclear" if the answer states a conclusion
        "experience_meets_minimum": string | null,       // same values as above
        "overall_meets_minimum": string | null           // overall conclusion stated in the answer
      },
      "financials": {
        "georgia_hs_head_coach_salary_range": string | null,            // e.g., "$100k–$200k" or textual range
        "georgia_salary_sources": string[],                              // URLs supporting Georgia HS salary range
        "ivy_league_oc_or_assistant_salary_range": string | null,        // e.g., "$120k–$300k" or textual range
        "ivy_salary_sources": string[],                                   // URLs supporting Ivy League OC/assistant salary range
        "compensation_change": string | null                              // "increase" / "decrease" / "depends"
      }
    }

    Extraction rules:
    - Extract only information explicitly stated in the answer.
    - For URLs: extract actual URLs present in the answer (plain or markdown). Do not invent URLs. Include full protocol (http/https).
    - If any item is missing in the answer, return null for that field and [] for missing URL lists.
    - Tenure should be expressed in seasons as-of the 2025 season; acceptable forms include "3 seasons", "third season (2025)", "since 2022 (4 seasons through 2025)", etc.
    - Official titles should be captured exactly as listed on sources (e.g., "Offensive Coordinator/Quarterbacks", "Associate Head Coach/Offensive Coordinator").
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _school_pretty_name(school_id: str) -> str:
    if school_id.lower() == "yale":
        return "Yale University"
    if school_id.lower() == "harvard":
        return "Harvard University"
    return school_id.capitalize()


def _safe(value: Optional[str]) -> str:
    return value or ""


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_school_oc(
    evaluator: Evaluator,
    parent_node,
    school_id: str,
    oc: Optional[OCInfo],
) -> None:
    """Build and verify the OC subtree for a given school."""
    pretty = _school_pretty_name(school_id)

    # Aggregator for this school's OC (critical)
    school_node = evaluator.add_parallel(
        id=f"{school_id}_offensive_coordinator",
        desc=f"{pretty} offensive coordinator identification and verification",
        parent=parent_node,
        critical=True,
    )

    # Ensure sources exist (critical existence)
    sources_exist = bool(oc and oc.sources and len(oc.sources) > 0)
    evaluator.add_custom_node(
        result=sources_exist,
        id=f"{school_id}_oc_references",
        desc=f"Provides at least one supporting URL for {pretty.split()[0]} OC identity/title/tenure from an official athletics page or other credible source",
        parent=school_node,
        critical=True,
    )

    # Name verification (critical leaf)
    name_leaf = evaluator.add_leaf(
        id=f"{school_id}_oc_name",
        desc=f"Provides the current {pretty.split()[0]} offensive coordinator's full name",
        parent=school_node,
        critical=True,
    )
    name_claim = f"The current offensive coordinator at {pretty} is '{_safe(oc.name)}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=(oc.sources if oc else []),
        additional_instruction=(
            "Verify on the provided official/credible sources (staff directory, press release, bio page, etc.) "
            "that this person is currently the offensive coordinator for the school. Accept variants like "
            "'Offensive Coordinator/Quarterbacks' or 'Associate Head Coach/Offensive Coordinator' as OC."
        ),
    )

    # Official title verification (critical leaf)
    title_leaf = evaluator.add_leaf(
        id=f"{school_id}_oc_official_title",
        desc="Provides and verifies the coach's official title (as listed by an official/credible source)",
        parent=school_node,
        critical=True,
    )
    title_claim = (
        f"On the cited sources, {_safe(oc.name)} is listed with the official title '{_safe(oc.official_title)}', "
        f"which is an offensive coordinator role at {pretty}."
    )
    await evaluator.verify(
        claim=title_claim,
        node=title_leaf,
        sources=(oc.sources if oc else []),
        additional_instruction=(
            "Check the exact title string on official pages. Minor formatting differences are acceptable, "
            "but the role must clearly denote offensive coordinator responsibilities."
        ),
    )

    # Tenure in seasons as-of 2025 (critical leaf)
    tenure_leaf = evaluator.add_leaf(
        id=f"{school_id}_oc_tenure_seasons_2025",
        desc="States how long the offensive coordinator has served in the current role, in seasons as of the 2025 season",
        parent=school_node,
        critical=True,
    )
    tenure_claim = (
        f"As of the 2025 season, {_safe(oc.name)} has served {_safe(oc.tenure_seasons_2025)} "
        f"in the offensive coordinator role at {pretty}."
    )
    await evaluator.verify(
        claim=tenure_claim,
        node=tenure_leaf,
        sources=(oc.sources if oc else []),
        additional_instruction=(
            "From the sources, infer start year and compute seasons through 2025. Accept phrasing like 'in his fourth season' "
            "or explicit enumeration (e.g., 2022, 2023, 2024, 2025 => 4 seasons). If sources are outdated or unclear, mark as not supported."
        ),
    )


async def verify_requirements_and_assessment(
    evaluator: Evaluator,
    parent_node,
    req: Optional[QualificationRequirements],
) -> None:
    """Build and verify the sequential subtree for requirements then candidate assessment."""
    seq_node = evaluator.add_sequential(
        id="qualification_requirements_and_assessment",
        desc="Identify typical minimum qualifications for OC roles (degree + years experience) and then assess the candidate against them",
        parent=parent_node,
        critical=True,
    )

    # 1) Standard requirements (parallel under sequential)
    std_node = evaluator.add_parallel(
        id="standard_qualification_requirements",
        desc="Typical minimum qualification requirements for offensive coordinator roles (Division I FCS / Ivy League): minimum degree and minimum years of coaching experience, with sources",
        parent=seq_node,
        critical=True,
    )

    # Degree references exist (critical existence)
    evaluator.add_custom_node(
        result=bool(req and req.degree_sources and len(req.degree_sources) > 0),
        id="minimum_degree_references",
        desc="Provides at least one supporting URL for the minimum degree requirement (e.g., job posting/career guidance source)",
        parent=std_node,
        critical=True,
    )

    # Minimum degree requirement claim (critical)
    deg_leaf = evaluator.add_leaf(
        id="minimum_degree_requirement",
        desc="Identifies the typical minimum educational degree required for such positions",
        parent=std_node,
        critical=True,
    )
    deg_claim = (
        f"The typical minimum educational degree requirement for Division I FCS/Ivy League offensive coordinator roles "
        f"is '{_safe(req.minimum_degree)}' (or equivalent)."
        if req else "The typical minimum educational degree requirement for Division I FCS/Ivy League offensive coordinator roles is ''."
    )
    await evaluator.verify(
        claim=deg_claim,
        node=deg_leaf,
        sources=(req.degree_sources if req else []),
        additional_instruction=(
            "Use the provided job postings, official HR requirements, or credible sources to verify the minimum educational degree. "
            "Accept reasonable equivalents if explicitly stated (e.g., 'Bachelor's required; Master's preferred')."
        ),
    )

    # Experience references exist (critical existence)
    evaluator.add_custom_node(
        result=bool(req and req.experience_sources and len(req.experience_sources) > 0),
        id="minimum_experience_references",
        desc="Provides at least one supporting URL for the minimum experience requirement (e.g., job posting/industry source)",
        parent=std_node,
        critical=True,
    )

    # Minimum experience requirement claim (critical)
    exp_leaf = evaluator.add_leaf(
        id="minimum_experience_requirement",
        desc="Identifies the typical minimum years of coaching experience required for such positions",
        parent=std_node,
        critical=True,
    )
    exp_claim = (
        f"The typical minimum years of coaching experience required for Division I FCS/Ivy League offensive coordinator roles "
        f"is '{_safe(req.minimum_experience_years)}'."
        if req else "The typical minimum years of coaching experience required for Division I FCS/Ivy League offensive coordinator roles is ''."
    )
    await evaluator.verify(
        claim=exp_claim,
        node=exp_leaf,
        sources=(req.experience_sources if req else []),
        additional_instruction=(
            "Prefer explicit numeric requirements if available (e.g., '5+ years'). If sources use qualitative phrasing (e.g., 'several years'), "
            "ensure the extracted value matches the source's language."
        ),
    )

    # 2) Candidate assessment (parallel, gated by sequential ordering)
    assess_node = evaluator.add_parallel(
        id="candidate_qualification_assessment",
        desc="Based on the identified standard requirements, assess whether the candidate (Master's degree, 6 years head coaching experience) meets the minimum qualifications",
        parent=seq_node,
        critical=True,
    )

    # Education meets minimum (critical)
    edu_leaf = evaluator.add_leaf(
        id="education_meets_minimum",
        desc="Concludes whether the candidate's education meets or exceeds the identified minimum degree requirement",
        parent=assess_node,
        critical=True,
    )
    edu_claim = (
        f"Given the identified minimum degree requirement '{_safe(req.minimum_degree)}', a candidate holding a Master's degree in Education "
        f"meets or exceeds the minimum educational requirement for Ivy League/Division I FCS offensive coordinator roles."
        if req else "Given the identified minimum degree requirement '', a candidate holding a Master's degree in Education meets or exceeds the minimum."
    )
    await evaluator.verify(
        claim=edu_claim,
        node=edu_leaf,
        additional_instruction=(
            "Judge this logically based on the minimum degree identified. If the minimum is Bachelor's, Master's meets/exceeds. "
            "If the minimum explicitly requires a specific major distinct from Education, only fail if sources require that specific major."
        ),
    )

    # Experience meets minimum (critical)
    exp_meet_leaf = evaluator.add_leaf(
        id="experience_meets_minimum",
        desc="Concludes whether the candidate's coaching experience meets or exceeds the identified minimum experience requirement",
        parent=assess_node,
        critical=True,
    )
    exp_meet_claim = (
        f"Given the identified minimum years of coaching experience '{_safe(req.minimum_experience_years)}', "
        f"a candidate with 6 years of head coaching experience meets or exceeds the minimum experience requirement for these roles."
        if req else "Given the identified minimum years of coaching experience '', a candidate with 6 years of head coaching experience meets or exceeds the minimum."
    )
    await evaluator.verify(
        claim=exp_meet_claim,
        node=exp_meet_leaf,
        additional_instruction=(
            "Unless the sources explicitly require college-level coaching experience, treat 'years of coaching experience' as inclusive of high school coaching. "
            "If sources explicitly require college coaching years, then evaluate accordingly."
        ),
    )

    # Overall conclusion (critical) – auto-gated by critical siblings via evaluator
    overall_leaf = evaluator.add_leaf(
        id="overall_meets_minimum_conclusion",
        desc="Provides a clear overall conclusion on whether the candidate meets minimum qualifications for these OC roles",
        parent=assess_node,
        critical=True,
    )
    overall_claim = (
        "Based on the identified minimum degree and experience requirements, and the candidate's Master's degree and 6 years of head coaching, "
        "the candidate overall meets the minimum qualifications for Ivy League/Division I FCS offensive coordinator roles."
    )
    await evaluator.verify(
        claim=overall_claim,
        node=overall_leaf,
        additional_instruction=(
            "This overall conclusion should be consistent with the prior two checks: only 'meets' if both education and experience meet/exceed the minimum."
        ),
    )


async def verify_financials(
    evaluator: Evaluator,
    parent_node,
    fin: Optional[FinancialImplications],
) -> None:
    """Build and verify the financial implications subtree."""
    fin_node = evaluator.add_parallel(
        id="financial_implications",
        desc="Salary ranges for (a) top Georgia high school head football coaches and (b) Ivy League OC/assistant coach roles, plus comparison direction, with sources",
        parent=parent_node,
        critical=True,
    )

    # Georgia HS salary references existence (critical)
    evaluator.add_custom_node(
        result=bool(fin and fin.georgia_salary_sources and len(fin.georgia_salary_sources) > 0),
        id="georgia_salary_references",
        desc="Provides at least one supporting URL for the Georgia high school coach salary range (salary database and/or credible journalism)",
        parent=fin_node,
        critical=True,
    )

    # Georgia HS salary range claim (critical)
    ga_salary_leaf = evaluator.add_leaf(
        id="georgia_hs_head_coach_salary_range",
        desc="Reports the typical salary range for top Georgia high school head football coaches",
        parent=fin_node,
        critical=True,
    )
    ga_salary_claim = (
        f"The typical salary range for top Georgia high school head football coaches is '{_safe(fin.georgia_hs_head_coach_salary_range)}'."
        if fin else "The typical salary range for top Georgia high school head football coaches is ''."
    )
    await evaluator.verify(
        claim=ga_salary_claim,
        node=ga_salary_leaf,
        sources=(fin.georgia_salary_sources if fin else []),
        additional_instruction=(
            "Verify the reported range on salary databases or credible journalism focused on Georgia HS football head coaches. "
            "Accept ranges reported across multiple credible sources."
        ),
    )

    # Ivy salary references existence (critical)
    evaluator.add_custom_node(
        result=bool(fin and fin.ivy_salary_sources and len(fin.ivy_salary_sources) > 0),
        id="ivy_salary_references",
        desc="Provides at least one supporting URL for Ivy League OC/assistant salary range(s) (salary database, official posting, and/or credible journalism)",
        parent=fin_node,
        critical=True,
    )

    # Ivy OC/assistant salary range claim (critical)
    ivy_salary_leaf = evaluator.add_leaf(
        id="ivy_league_coordinator_or_assistant_salary_range",
        desc="Reports the approximate salary range(s) for Ivy League offensive coordinator and/or assistant coach positions",
        parent=fin_node,
        critical=True,
    )
    ivy_salary_claim = (
        f"The approximate salary range(s) for Ivy League offensive coordinator or assistant coach positions is '{_safe(fin.ivy_league_oc_or_assistant_salary_range)}'."
        if fin else "The approximate salary range(s) for Ivy League offensive coordinator or assistant coach positions is ''."
    )
    await evaluator.verify(
        claim=ivy_salary_claim,
        node=ivy_salary_leaf,
        sources=(fin.ivy_salary_sources if fin else []),
        additional_instruction=(
            "Verify the reported range on credible sources (salary databases, official postings, or reputable journalism). "
            "Accept reasonable variation across institutions as long as the stated range is supported."
        ),
    )

    # Compensation change comparison (critical)
    comp_leaf = evaluator.add_leaf(
        id="compensation_change_comparison",
        desc="Briefly compares the two salary ranges and states whether the transition is likely an increase or decrease in compensation",
        parent=fin_node,
        critical=True,
    )
    comp_claim = (
        f"Based on the reported Georgia HS head coach range '{_safe(fin.georgia_hs_head_coach_salary_range)}' versus "
        f"Ivy League OC/assistant range '{_safe(fin.ivy_league_oc_or_assistant_salary_range)}', the transition would likely result in "
        f"a '{_safe(fin.compensation_change)}' in compensation."
        if fin else "Based on the reported ranges '' versus '', the transition would likely result in a ''."
    )
    await evaluator.verify(
        claim=comp_claim,
        node=comp_leaf,
        additional_instruction=(
            "Judge the direction (increase/decrease/depends) logically from the two ranges as presented in the answer. "
            "If ranges overlap or are inconclusive, 'depends' is acceptable."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the Ivy League offensive coordinator career planning task.
    """
    # Initialize evaluator (root is non-critical by design; we add a critical top-level node)
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

    # Add top-level critical aggregator matching rubric
    top_node = evaluator.add_parallel(
        id="career_transition_evaluation",
        desc="Career planning analysis for transitioning from GA high school head coach to Ivy League offensive coordinator roles (Yale, Harvard)",
        parent=root,
        critical=True,
    )

    # Extract structured info
    extraction = await evaluator.extract(
        prompt=prompt_extract_career_planning(),
        template_class=CareerPlanningExtraction,
        extraction_name="career_planning_extraction",
    )

    # Record candidate profile as custom info for clarity
    evaluator.add_custom_info(
        info={
            "candidate_education": "Master's degree in Education",
            "candidate_experience_years": "6 years (head coaching)",
            "focus_schools": ["Yale University", "Harvard University"]
        },
        info_type="candidate_profile"
    )

    # Section 1: Current Offensive Coordinators (critical aggregator)
    coc_node = evaluator.add_parallel(
        id="current_offensive_coordinators",
        desc="For both Yale and Harvard: identify current offensive coordinator name, verify official title, provide tenure in seasons as of 2025, and include supporting URLs",
        parent=top_node,
        critical=True,
    )
    # Build Yale and Harvard OC verification subtrees
    await verify_school_oc(evaluator, coc_node, "yale", extraction.yale)
    await verify_school_oc(evaluator, coc_node, "harvard", extraction.harvard)

    # Section 2: Qualification Requirements & Assessment (sequential and critical)
    await verify_requirements_and_assessment(evaluator, top_node, extraction.requirements)

    # Section 3: Financial Implications (critical)
    await verify_financials(evaluator, top_node, extraction.financials)

    # Return structured summary
    return evaluator.get_summary()