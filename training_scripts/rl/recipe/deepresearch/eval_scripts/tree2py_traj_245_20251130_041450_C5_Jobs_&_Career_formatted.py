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
TASK_ID = "power4_head_coach_profile"
TASK_DESCRIPTION = (
    "You are a career advisor helping a college football coordinator who wants to become a Power 4 head coach. "
    "Create a career advancement profile that includes: (1) the educational credentials required for Power 4 head "
    "coaching positions, specifying both bachelor's and master's degree expectations and typical fields of study, "
    "(2) the typical career progression path from coordinator to head coach, including the role of coordinator positions "
    "in this progression, (3) current market salary ranges for both top Power 4 coordinators and top Power 4 head coaches, "
    "and (4) detailed information about one specific current Power 4 head coaching vacancy, including the school name, "
    "the date when the vacancy opened, and which Power 4 conference the school belongs to. For each major section of your "
    "profile (educational credentials, career progression, salary information, and the specific vacancy), provide at least "
    "one reference URL that supports the information."
)

ALLOWED_VACANCY_SCHOOLS = [
    "LSU",
    "Penn State",
    "Florida",
    "Auburn",
    "Oklahoma State",
    "Arkansas",
    "Stanford",
    "UCLA",
    "California",
]

SCHOOL_SYNONYMS: Dict[str, List[str]] = {
    "LSU": ["Louisiana State", "Louisiana State University", "LSU Tigers"],
    "Penn State": ["Pennsylvania State", "Pennsylvania State University", "PSU", "Penn State Nittany Lions"],
    "Florida": ["University of Florida", "UF", "Florida Gators"],
    "Auburn": ["Auburn University", "AU"],
    "Oklahoma State": ["Oklahoma State University", "OSU Cowboys", "OSU (Oklahoma State)"],
    "Arkansas": ["University of Arkansas", "Arkansas Razorbacks", "UA (Arkansas)"],
    "Stanford": ["Stanford University", "Stanford Cardinal"],
    "UCLA": ["University of California, Los Angeles", "UCLA Bruins"],
    "California": ["Cal", "University of California", "University of California, Berkeley", "UC Berkeley", "California Golden Bears", "Cal (Berkeley)"],
}


def canonicalize_school(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = name.strip().lower()
    for canonical in ALLOWED_VACANCY_SCHOOLS:
        if n == canonical.lower():
            return canonical
        for syn in SCHOOL_SYNONYMS.get(canonical, []):
            if n == syn.strip().lower():
                return canonical
    return None


def is_valid_power4_vacancy_school(name: Optional[str]) -> bool:
    return canonicalize_school(name) is not None


def _comma_join(items: Optional[List[str]]) -> str:
    try:
        if not items:
            return ""
        return ", ".join([i for i in items if isinstance(i, str) and i.strip() != ""])
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EducationInfo(BaseModel):
    bachelors_requirement_statement: Optional[str] = None
    bachelors_fields: List[str] = Field(default_factory=list)
    masters_requirement_statement: Optional[str] = None
    masters_fields: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class CareerPathInfo(BaseModel):
    coordinator_role_statement: Optional[str] = None
    progression_sequence_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class SalaryInfo(BaseModel):
    coordinator_salary_range_text: Optional[str] = None
    head_coach_salary_range_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class VacancyInfo(BaseModel):
    school_name: Optional[str] = None
    opening_date_text: Optional[str] = None
    conference: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ProfileExtraction(BaseModel):
    education: Optional[EducationInfo] = None
    progression: Optional[CareerPathInfo] = None
    salary: Optional[SalaryInfo] = None
    vacancy: Optional[VacancyInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_profile() -> str:
    return (
        "Extract a structured career advancement profile from the answer, organized into four sections: "
        "education, progression, salary, and vacancy. "
        "Return a JSON object with keys: education, progression, salary, vacancy.\n\n"
        "education:\n"
        "- bachelors_requirement_statement: The text statement about bachelor's degree expectation or requirement for college football head coaching roles.\n"
        "- bachelors_fields: List of typical bachelor's degree fields (e.g., sports management, physical education, kinesiology, exercise science, or related fields).\n"
        "- masters_requirement_statement: The text statement about master's degree being typically required or strongly preferred.\n"
        "- masters_fields: List of typical master's degree fields (e.g., sports management, education administration, athletic administration).\n"
        "- urls: All URLs cited that specifically support educational requirements.\n\n"
        "progression:\n"
        "- coordinator_role_statement: Statement that offensive/defensive coordinator roles are senior assistant positions that often precede head coaching opportunities.\n"
        "- progression_sequence_text: Text description of typical progression path (e.g., graduate assistant → position coach → coordinator → head coach).\n"
        "- urls: All URLs cited that support career progression paths.\n\n"
        "salary:\n"
        "- coordinator_salary_range_text: The salary range stated for top college football coordinators (e.g., ~$2M–$3.1M).\n"
        "- head_coach_salary_range_text: The salary range stated for top Power 4 head coaches (e.g., ~$11M–$13M+).\n"
        "- urls: All URLs cited that support the salary figures.\n\n"
        "vacancy:\n"
        "- school_name: The school selected for the current Power 4 head coaching vacancy.\n"
        "- opening_date_text: The stated date when the vacancy opened/was announced.\n"
        "- conference: The stated Power 4 conference of the school (SEC, Big Ten, Big 12, ACC).\n"
        "- urls: All URLs cited that provide information about this specific vacancy.\n\n"
        "Special URL extraction rules: only include actual URLs present in the answer; include full URLs with protocol if available."
    )


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_education(evaluator: Evaluator, parent_node, edu: Optional[EducationInfo]) -> None:
    edu_node = evaluator.add_parallel(
        id="Educational_Credentials",
        desc="Accurate description of educational requirements for Power 4 head coaching positions",
        parent=parent_node,
        critical=True,
    )

    urls = edu.urls if edu else []
    # Leaf: Education_Reference_URL (existence)
    evaluator.add_custom_node(
        result=(len(urls) >= 1),
        id="Education_Reference_URL",
        desc="Provides at least one valid URL source documenting educational requirements for college football coaches",
        parent=edu_node,
        critical=True,
    )

    # Leaf: Bachelors_Degree_Requirement
    bachelors_leaf = evaluator.add_leaf(
        id="Bachelors_Degree_Requirement",
        desc="States that a bachelor's degree is required for college coaching positions, typically in sports management, physical education, kinesiology, exercise science, or related field",
        parent=edu_node,
        critical=True,
    )
    bachelors_fields_text = _comma_join(edu.bachelors_fields if edu else [])
    bachelors_stmt = (edu.bachelors_requirement_statement or "").strip() if edu else ""
    bachelors_claim = (
        "College football head coaching roles (and college coaching roles generally) require a bachelor's degree; "
        f"typical fields include {bachelors_fields_text if bachelors_fields_text else 'sports management or related fields'}. "
        f"The answer's statement is: '{bachelors_stmt}'."
    )
    await evaluator.verify(
        claim=bachelors_claim,
        node=bachelors_leaf,
        sources=urls,
        additional_instruction=(
            "Judge whether at least one provided source explicitly supports that a bachelor's degree is required for college coaching/head coaching roles, "
            "and that typical fields include sports-related disciplines (sports management, physical education, kinesiology, exercise science, or similar). "
            "Allow reasonable wording variations."
        ),
    )

    # Leaf: Masters_Degree_Requirement
    masters_leaf = evaluator.add_leaf(
        id="Masters_Degree_Requirement",
        desc="States that a master's degree is typically required or strongly preferred for college head coaching positions, often in sports management, education administration, or athletic administration",
        parent=edu_node,
        critical=True,
    )
    masters_fields_text = _comma_join(edu.masters_fields if edu else [])
    masters_stmt = (edu.masters_requirement_statement or "").strip() if edu else ""
    masters_claim = (
        "For college football head coaching roles, a master's degree is typically required or strongly preferred; "
        f"common master's fields include {masters_fields_text if masters_fields_text else 'sports management, education/athletic administration, or similar'}. "
        f"The answer's statement is: '{masters_stmt}'."
    )
    await evaluator.verify(
        claim=masters_claim,
        node=masters_leaf,
        sources=urls,
        additional_instruction=(
            "Check whether at least one source indicates that a master's degree is commonly expected or strongly preferred for college head coaches, "
            "and mentions relevant fields (sports management, education administration, athletic administration, etc.). "
            "Allow general phrasing that implies strong preference."
        ),
    )


async def verify_progression(evaluator: Evaluator, parent_node, prog: Optional[CareerPathInfo]) -> None:
    prog_node = evaluator.add_parallel(
        id="Career_Progression_Path",
        desc="Accurate description of typical career path from coordinator to head coach",
        parent=parent_node,
        critical=True,
    )

    urls = prog.urls if prog else []
    # Leaf: Experience_Reference_URL (existence)
    evaluator.add_custom_node(
        result=(len(urls) >= 1),
        id="Experience_Reference_URL",
        desc="Provides at least one valid URL source documenting career progression paths for football coaches",
        parent=prog_node,
        critical=True,
    )

    # Leaf: Coordinator_Level_Position
    coord_leaf = evaluator.add_leaf(
        id="Coordinator_Level_Position",
        desc="Identifies that coordinator positions (offensive coordinator or defensive coordinator) are senior assistant roles that typically precede head coaching opportunities",
        parent=prog_node,
        critical=True,
    )
    coord_stmt = (prog.coordinator_role_statement or "").strip() if prog else ""
    coord_claim = (
        "Offensive and defensive coordinator roles are senior assistant positions that typically precede head coaching opportunities. "
        f"The answer's statement is: '{coord_stmt}'."
    )
    await evaluator.verify(
        claim=coord_claim,
        node=coord_leaf,
        sources=urls,
        additional_instruction=(
            "Verify that the provided source(s) describe coordinator roles as senior assistant positions and indicate that coordinator experience "
            "commonly leads to head coaching opportunities. Allow synonyms (e.g., 'OC/DC', 'senior staff')."
        ),
    )

    # Leaf: Progression_Sequence
    seq_leaf = evaluator.add_leaf(
        id="Progression_Sequence",
        desc="Describes the typical progression path including positions such as graduate assistant, position coach, coordinator, and head coach",
        parent=prog_node,
        critical=True,
    )
    seq_stmt = (prog.progression_sequence_text or "").strip() if prog else ""
    seq_claim = (
        "The typical coaching progression path includes: graduate assistant → position coach → coordinator → head coach. "
        f"The answer's described sequence is: '{seq_stmt}'."
    )
    await evaluator.verify(
        claim=seq_claim,
        node=seq_leaf,
        sources=urls,
        additional_instruction=(
            "Confirm that at least one provided source lays out a pathway that reasonably matches GA to position coach to coordinator to head coach. "
            "Allow minor variations in titles or intermediate steps."
        ),
    )


async def verify_salary(evaluator: Evaluator, parent_node, sal: Optional[SalaryInfo]) -> None:
    sal_node = evaluator.add_parallel(
        id="Salary_Market_Analysis",
        desc="Accurate salary range information for both coordinator and head coach positions at Power 4 schools",
        parent=parent_node,
        critical=True,
    )

    urls = sal.urls if sal else []
    # Leaf: Salary_Reference_URL (existence)
    evaluator.add_custom_node(
        result=(len(urls) >= 1),
        id="Salary_Reference_URL",
        desc="Provides at least one valid URL source documenting salary information for college football coaches",
        parent=sal_node,
        critical=True,
    )

    # Leaf: Coordinator_Salary_Range
    coord_sal_leaf = evaluator.add_leaf(
        id="Coordinator_Salary_Range",
        desc="Provides accurate salary range for top college football coordinators, indicating ranges of approximately $2 million to $3.1 million annually for top coordinators",
        parent=sal_node,
        critical=True,
    )
    coord_text = (sal.coordinator_salary_range_text or "").strip() if sal else ""
    coord_claim = (
        "Top Power 4 football coordinators earn approximately $2 million to $3.1 million per year. "
        f"The answer's stated range/wording is: '{coord_text}'."
    )
    await evaluator.verify(
        claim=coord_claim,
        node=coord_sal_leaf,
        sources=urls,
        additional_instruction=(
            "Use the cited pages (e.g., salary databases or reports) to verify that top coordinators' annual compensation is roughly in the $2M–$3.1M range. "
            "Allow reasonable rounding and inclusion/exclusion of bonuses when clearly indicated."
        ),
    )

    # Leaf: Head_Coach_Salary_Range
    hc_sal_leaf = evaluator.add_leaf(
        id="Head_Coach_Salary_Range",
        desc="Provides accurate salary range for Power 4 college head coaches, indicating top coaches earn approximately $11 million to $13 million+ annually",
        parent=sal_node,
        critical=True,
    )
    hc_text = (sal.head_coach_salary_range_text or "").strip() if sal else ""
    hc_claim = (
        "Top Power 4 head coaches earn approximately $11 million to $13 million+ per year. "
        f"The answer's stated range/wording is: '{hc_text}'."
    )
    await evaluator.verify(
        claim=hc_claim,
        node=hc_sal_leaf,
        sources=urls,
        additional_instruction=(
            "Verify using the cited sources (e.g., contract databases or news articles) that top-tier Power 4 head coaches' annual compensation "
            "is in the ~$11M–$13M+ range. Allow reasonable rounding and that '+' indicates above $13M for some coaches."
        ),
    )


async def verify_vacancy(evaluator: Evaluator, parent_node, vac: Optional[VacancyInfo]) -> None:
    vac_node = evaluator.add_parallel(
        id="Target_Vacancy_Profile",
        desc="Information about one specific current Power 4 head coaching vacancy",
        parent=parent_node,
        critical=True,
    )

    urls = vac.urls if vac else []
    # Leaf: Vacancy_Reference_URL (existence)
    evaluator.add_custom_node(
        result=(len(urls) >= 1),
        id="Vacancy_Reference_URL",
        desc="Provides at least one valid URL source with information about the specific coaching vacancy",
        parent=vac_node,
        critical=True,
    )

    # Leaf: Valid_Power4_School (membership check against allowed list/synonyms)
    valid_school_result = is_valid_power4_vacancy_school(vac.school_name if vac else None)
    evaluator.add_custom_node(
        result=valid_school_result,
        id="Valid_Power4_School",
        desc="Identifies one of the current Power 4 head coaching vacancies: LSU, Penn State, Florida, Auburn, Oklahoma State, Arkansas, Stanford, UCLA, or California",
        parent=vac_node,
        critical=True,
    )

    # Leaf: Opening_Date (verified by URLs)
    open_leaf = evaluator.add_leaf(
        id="Opening_Date",
        desc="Provides the date when the coaching vacancy was announced or opened (e.g., LSU opened October 26, 2025)",
        parent=vac_node,
        critical=True,
    )
    school_display = vac.school_name or ""
    open_text = (vac.opening_date_text or "").strip() if vac else ""
    open_claim = (
        f"A head coaching vacancy at {school_display} was announced/opened on {open_text}."
    )
    await evaluator.verify(
        claim=open_claim,
        node=open_leaf,
        sources=urls,
        additional_instruction=(
            "Confirm that the provided source(s) explicitly state the announcement/opening date for the head coaching vacancy at the named school. "
            "Allow minor date format variations; the date should clearly correspond to the announcement/opening."
        ),
    )

    # Leaf: Conference_Affiliation (verified by URLs)
    conf_leaf = evaluator.add_leaf(
        id="Conference_Affiliation",
        desc="Correctly identifies which Power 4 conference the school belongs to (SEC, Big Ten, Big 12, or ACC)",
        parent=vac_node,
        critical=True,
    )
    conf_text = (vac.conference or "").strip() if vac else ""
    conf_claim = (
        f"{school_display} belongs to the {conf_text} conference (one of SEC, Big Ten, Big 12, ACC)."
    )
    await evaluator.verify(
        claim=conf_claim,
        node=conf_leaf,
        sources=urls,
        additional_instruction=(
            "Check the provided source(s) for explicit mention of the school's current Power 4 conference affiliation (SEC, Big Ten, Big 12, or ACC). "
            "Allow common naming variants (e.g., 'Southeastern Conference' for SEC)."
        ),
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
    Evaluate the Power 4 head coach career advancement profile answer.
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
        default_model=model,
    )

    # Extract the structured profile from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_profile(),
        template_class=ProfileExtraction,
        extraction_name="career_advancement_profile",
    )

    # Top-level critical node representing the whole profile
    profile_node = evaluator.add_parallel(
        id="Career_Advancement_Profile",
        desc="Complete career advancement profile including educational requirements, career progression path, salary market analysis, and target vacancy information",
        parent=root,
        critical=True,
    )

    # Optional: record allowed schools for transparency in summary
    evaluator.add_ground_truth({
        "allowed_vacancy_schools": ALLOWED_VACANCY_SCHOOLS,
        "school_synonyms": SCHOOL_SYNONYMS,
    }, gt_type="allowed_vacancy_list")

    # Build and verify each major section
    await verify_education(evaluator, profile_node, extracted.education)
    await verify_progression(evaluator, profile_node, extracted.progression)
    await verify_salary(evaluator, profile_node, extracted.salary)
    await verify_vacancy(evaluator, profile_node, extracted.vacancy)

    # Return final summary
    return evaluator.get_summary()