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
TASK_ID = "setonhall_psych_awards_2023"
TASK_DESCRIPTION = (
    "Identify the tenured Associate Professor of Psychology at Seton Hall University who received both the College of Arts and "
    "Science Researcher of the Year award and the University Faculty Researcher of the Year award in 2023. Once you have identified "
    "this professor, provide the following information: (1) The name of the university where this professor earned their PhD degree "
    "in 2016, (2) The city where that PhD-granting university is located, (3) The name of the institution where this professor held a "
    "Visiting Assistant Professor position immediately before joining Seton Hall University (during the period 2014-2016), (4) The U.S. "
    "state where that previous employment institution is located, and (5) A reference URL that confirms this professor's employment at "
    "the institution mentioned in item 3."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProfessorData(BaseModel):
    # Identification and role at Seton Hall
    professor_name: Optional[str] = None
    seton_hall_role: Optional[str] = None  # e.g., "tenured Associate Professor of Psychology"
    awards_2023: List[str] = Field(default_factory=list)  # Names of awards claimed received in 2023
    sources_general: List[str] = Field(default_factory=list)  # All URLs cited in the answer

    # PhD information
    phd_institution: Optional[str] = None
    phd_year: Optional[str] = None  # Use string to allow variants like "2016"
    phd_city: Optional[str] = None
    phd_sources: List[str] = Field(default_factory=list)

    # Previous employment immediately before SHU (2014-2016)
    previous_employment_institution: Optional[str] = None
    previous_employment_role: Optional[str] = None  # e.g., "Visiting Assistant Professor"
    previous_employment_period: Optional[str] = None  # e.g., "2014-2016"
    previous_employment_state: Optional[str] = None
    previous_employment_reference_url: Optional[str] = None  # A single URL that confirms employment
    prev_employment_sources: List[str] = Field(default_factory=list)  # Additional URLs related to the employment


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_professor_data() -> str:
    return """
    Extract the requested structured information from the answer about the Seton Hall University professor and related details.
    Return a JSON object with the following fields:

    1. professor_name: The full name of the professor identified.
    2. seton_hall_role: The role/title at Seton Hall University, including whether they are tenured and that they are an Associate Professor of Psychology (if stated).
    3. awards_2023: A list of award names the answer claims the professor received in 2023. Include text as-is (e.g., "College of Arts and Sciences Researcher of the Year", "University Faculty Researcher of the Year").
    4. sources_general: Extract ALL URLs present in the answer (including profile pages, news pages, CVs, etc.). If none, return an empty array.

    PhD information:
    5. phd_institution: The university where the professor earned their PhD degree.
    6. phd_year: The year the PhD degree was earned. It should be "2016" if the answer claims so; return whatever the answer states.
    7. phd_city: The city where the PhD-granting university is located (as stated or implied in the answer).
    8. phd_sources: URLs cited in the answer that specifically support the PhD institution/year/city. If none, return an empty array.

    Previous employment (immediately before Seton Hall, during 2014–2016):
    9. previous_employment_institution: The institution where the professor held a Visiting Assistant Professor position immediately prior to joining Seton Hall.
    10. previous_employment_role: The role/title at that institution (e.g., "Visiting Assistant Professor").
    11. previous_employment_period: The time period for that position (e.g., "2014-2016").
    12. previous_employment_state: The U.S. state where that institution is located.
    13. previous_employment_reference_url: A single most-direct URL (if provided) that confirms the professor's employment at the institution in item 9.
    14. prev_employment_sources: Any additional URLs cited in the answer that support or relate to the previous employment. If none, return an empty array.

    IMPORTANT:
    - Extract only what the answer explicitly states; do not invent.
    - For URLs, include the full absolute URL. Accept URLs in markdown links or plain text. If a URL lacks protocol, prepend 'http://'.
    - If any field is missing from the answer, return null for that field; for arrays, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def merge_sources(*lists: List[str], single_url: Optional[str] = None) -> List[str]:
    """Merge multiple lists of URLs and an optional single URL into a unique list."""
    merged: List[str] = []
    for lst in lists:
        if lst:
            merged.extend(lst)
    if single_url:
        merged.append(single_url)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for url in merged:
        if url and url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def str_present(s: Optional[str]) -> bool:
    return bool(s and s.strip())


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_professor_identification(evaluator: Evaluator, root_node, data: ProfessorData) -> None:
    """
    Node: Professor_Identification (critical under critical root)
    - Existence of professor_name
    - Verify role/title (tenured Associate Professor of Psychology at Seton Hall University)
    - Verify both awards in 2023
    """
    node = evaluator.add_parallel(
        id="Professor_Identification",
        desc=("Correctly identify the tenured Associate Professor of Psychology at Seton Hall University who received both the "
              "College of Arts and Science Researcher of the Year award and the University Faculty Researcher of the Year award in 2023"),
        parent=root_node,
        critical=True  # Must be critical because parent root is critical
    )

    # Existence check for professor name
    evaluator.add_custom_node(
        result=str_present(data.professor_name),
        id="professor_name_present",
        desc="Professor name is provided",
        parent=node,
        critical=True
    )

    # Verify role/title at Seton Hall University (includes 'tenured' and 'Associate Professor of Psychology')
    role_leaf = evaluator.add_leaf(
        id="role_title_verified",
        desc="Professor is a tenured Associate Professor of Psychology at Seton Hall University",
        parent=node,
        critical=True
    )
    role_claim = (
        f"The professor named '{data.professor_name or ''}' is a tenured Associate Professor of Psychology at Seton Hall University."
    )
    await evaluator.verify(
        claim=role_claim,
        node=role_leaf,
        sources=merge_sources(data.sources_general),
        additional_instruction=(
            "Verify that the sources confirm both: (1) the person is an Associate Professor of Psychology at Seton Hall University, "
            "(2) the person is tenured. Allow minor variations such as 'Dept. of Psychology', abbreviations, or presence/absence of middle initials."
        )
    )

    # Verify awards: both awards in 2023
    awards_leaf = evaluator.add_leaf(
        id="awards_2023_verified",
        desc=("Professor received both 'College of Arts and Science(s) Researcher of the Year' and "
              "'University Faculty Researcher of the Year' awards in 2023"),
        parent=node,
        critical=True
    )
    awards_claim = (
        f"The professor named '{data.professor_name or ''}' received both the College of Arts and Science(s) Researcher of the Year "
        f"award and the University Faculty Researcher of the Year award in 2023 at Seton Hall University."
    )
    await evaluator.verify(
        claim=awards_claim,
        node=awards_leaf,
        sources=merge_sources(data.sources_general),
        additional_instruction=(
            "Confirm that the person received BOTH awards in the year 2023. Accept 'College of Arts and Sciences (CAS) Researcher of the Year' "
            "as equivalent to 'College of Arts and Science Researcher of the Year'. The pages should clearly indicate 2023 and the two award titles."
        )
    )


async def verify_phd_institution(evaluator: Evaluator, root_node, data: ProfessorData) -> None:
    """
    Node: PhD_Institution
    - Existence of phd_institution and year
    - Verify PhD institution and year 2016
    """
    node = evaluator.add_parallel(
        id="PhD_Institution",
        desc="Correctly identify the university where the professor earned their PhD degree in 2016",
        parent=root_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=str_present(data.phd_institution) and str_present(data.phd_year),
        id="phd_institution_present",
        desc="PhD institution and year are provided",
        parent=node,
        critical=True
    )

    phd_leaf = evaluator.add_leaf(
        id="phd_institution_verified",
        desc="The professor earned their PhD degree in 2016 from the identified university",
        parent=node,
        critical=True
    )
    phd_claim = (
        f"The professor earned a PhD degree in 2016 from {data.phd_institution or ''}."
    )
    await evaluator.verify(
        claim=phd_claim,
        node=phd_leaf,
        sources=merge_sources(data.phd_sources, data.sources_general),
        additional_instruction=(
            "Verify that the referenced page(s) explicitly state the PhD degree was awarded in 2016 and the awarding university matches the provided name. "
            "Minor name variations (e.g., official vs. abbreviated names) are acceptable."
        )
    )


async def verify_phd_city(evaluator: Evaluator, root_node, data: ProfessorData) -> None:
    """
    Node: PhD_Institution_City
    - Existence of phd_city
    - Verify city of PhD institution
    """
    node = evaluator.add_parallel(
        id="PhD_Institution_City",
        desc="Correctly identify the city where the PhD-granting university is located",
        parent=root_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=str_present(data.phd_city),
        id="phd_city_present",
        desc="PhD institution city is provided",
        parent=node,
        critical=True
    )

    city_leaf = evaluator.add_leaf(
        id="phd_city_verified",
        desc="The city where the PhD-granting university is located is correctly identified",
        parent=node,
        critical=True
    )
    city_claim = (
        f"The city where {data.phd_institution or 'the PhD-granting university'} is located is {data.phd_city or ''}."
    )
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=merge_sources(data.phd_sources, data.sources_general),
        additional_instruction=(
            "Verify the city associated with the university (main campus or the campus explicitly stated for the PhD degree). "
            "If the source clearly indicates a campus location different from the main city, accept that campus city as correct."
        )
    )


async def verify_previous_employment(evaluator: Evaluator, root_node, data: ProfessorData) -> None:
    """
    Node: Previous_Employment
    - Existence of previous employment institution
    - Verify Visiting Assistant Professor position at that institution during 2014-2016
    """
    node = evaluator.add_parallel(
        id="Previous_Employment",
        desc=("Correctly identify the institution where the professor held a Visiting Assistant Professor position immediately before "
              "joining Seton Hall University (2014-2016)"),
        parent=root_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=str_present(data.previous_employment_institution),
        id="prev_employment_institution_present",
        desc="Previous employment institution is provided",
        parent=node,
        critical=True
    )

    prev_leaf = evaluator.add_leaf(
        id="prev_employment_verified",
        desc="The Visiting Assistant Professor position (2014-2016) at the identified institution is correctly cited",
        parent=node,
        critical=True
    )

    prev_sources = merge_sources(
        data.prev_employment_sources,
        data.sources_general,
        single_url=data.previous_employment_reference_url
    )
    prev_claim = (
        f"Immediately before joining Seton Hall University, during 2014-2016, the professor held a Visiting Assistant Professor "
        f"position at {data.previous_employment_institution or ''}."
    )
    await evaluator.verify(
        claim=prev_claim,
        node=prev_leaf,
        sources=prev_sources,
        additional_instruction=(
            "Confirm that the sources clearly indicate the person held a 'Visiting Assistant Professor' or equivalent visiting appointment at the named institution "
            "during 2014–2016, and that this employment was immediately prior to joining Seton Hall University."
        )
    )


async def verify_previous_employment_state(evaluator: Evaluator, root_node, data: ProfessorData) -> None:
    """
    Node: Previous_Employment_State
    - Existence of previous employment state
    - Verify the U.S. state of that institution
    """
    node = evaluator.add_parallel(
        id="Previous_Employment_State",
        desc="Correctly identify the U.S. state where the previous employment institution is located",
        parent=root_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=str_present(data.previous_employment_state),
        id="prev_state_present",
        desc="Previous employment state is provided",
        parent=node,
        critical=True
    )

    state_leaf = evaluator.add_leaf(
        id="prev_state_verified",
        desc="The U.S. state of the previous employment institution is correctly identified",
        parent=node,
        critical=True
    )

    prev_sources = merge_sources(
        data.prev_employment_sources,
        data.sources_general,
        single_url=data.previous_employment_reference_url
    )
    state_claim = (
        f"The institution '{data.previous_employment_institution or ''}' is located in the U.S. state of {data.previous_employment_state or ''}."
    )
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=prev_sources,
        additional_instruction=(
            "Verify that the institution's location corresponds to the stated U.S. state. Accept official pages, university directories, "
            "or reputable sources that clearly list the state."
        )
    )


async def verify_reference_url(evaluator: Evaluator, root_node, data: ProfessorData) -> None:
    """
    Node: Reference_URL
    - Existence of a specific reference URL
    - Verify that the URL confirms the professor's employment at the previous institution
    """
    node = evaluator.add_parallel(
        id="Reference_URL",
        desc=("Provide a valid reference URL that confirms the professor's employment at the institution identified in the previous step"),
        parent=root_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=str_present(data.previous_employment_reference_url),
        id="ref_url_present",
        desc="Reference URL for previous employment is provided",
        parent=node,
        critical=True
    )

    ref_leaf = evaluator.add_leaf(
        id="ref_url_confirms_employment",
        desc="The provided reference URL confirms the professor's Visiting Assistant Professor employment (2014-2016) at the institution",
        parent=node,
        critical=True
    )

    ref_claim = (
        f"The webpage confirms that {data.professor_name or 'the professor'} held a Visiting Assistant Professor position at "
        f"{data.previous_employment_institution or 'the institution'} during 2014-2016."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=data.previous_employment_reference_url,
        additional_instruction=(
            "Check that the single provided URL explicitly supports the employment claim (role and period). Prefer an official page, CV, faculty directory, "
            "or reputable source that clearly states the appointment."
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
    Evaluate the answer for the Seton Hall Psychology awards 2023 task.
    Builds a sequential critical root with six critical sub-nodes (each with internal critical leaf checks).
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Sequential since later steps depend on correct identification
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify the correct professor and complete the sequential verification task",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Mark root as critical to enforce mandatory correctness; Note: root node is created non-critical by initialize,
    # so we update its properties to align with rubric
    root.critical = True
    root.desc = "Identify the correct professor and complete the sequential verification task"

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_professor_data(),
        template_class=ProfessorData,
        extraction_name="professor_data"
    )

    # Build and verify each step (critical children under critical root)
    await verify_professor_identification(evaluator, root, extracted)
    await verify_phd_institution(evaluator, root, extracted)
    await verify_phd_city(evaluator, root, extracted)
    await verify_previous_employment(evaluator, root, extracted)
    await verify_previous_employment_state(evaluator, root, extracted)
    await verify_reference_url(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()