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
TASK_ID = "pa_career_services_exec_director"
TASK_DESCRIPTION = (
    "Identify the career services executive director at a university in Pennsylvania who meets all of the following criteria: "
    "(1) Has been serving in their current executive director or director-level role since 2018 or earlier, "
    "(2) Joined their current institution's career services office in 1998 or earlier, "
    "(3) Holds a master's degree in Counseling from Shippensburg University, "
    "(4) Holds an EdD in Higher Education Administration from the same university where they currently work, and "
    "(5) Completed an undergraduate degree with a major in Psychology. Provide the person's full name and the name of the university where they work."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class PersonExtraction(BaseModel):
    full_name: Optional[str] = None
    university_name: Optional[str] = None
    role_title: Optional[str] = None

    current_role_start_info: Optional[str] = None  # e.g., "since 2014", "appointed 2012"
    joined_career_services_info: Optional[str] = None  # e.g., "joined in 1996", "with the office since 1995"

    masters_degree_title: Optional[str] = None  # e.g., "Master of Science in Counseling"
    masters_field: Optional[str] = None  # e.g., "Counseling"
    masters_institution: Optional[str] = None  # e.g., "Shippensburg University"

    edd_degree_title: Optional[str] = None  # e.g., "Ed.D. in Higher Education Administration"
    edd_field: Optional[str] = None  # e.g., "Higher Education Administration"
    edd_institution: Optional[str] = None  # University awarding the EdD

    undergraduate_degree_title: Optional[str] = None  # e.g., "B.A. in Psychology"
    undergraduate_major: Optional[str] = None  # e.g., "Psychology"

    person_profile_url: Optional[str] = None  # direct profile/bio page URL if provided
    university_homepage_url: Optional[str] = None  # university homepage or "about" page if provided
    source_urls: List[str] = Field(default_factory=list)  # all other URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_person_info() -> str:
    return (
        "From the provided answer, extract the following fields exactly as stated:\n"
        "1) full_name: The person's full name.\n"
        "2) university_name: The name of the university where the person works.\n"
        "3) role_title: The person's role title (e.g., 'Executive Director of Career Services', 'Director of Career Services').\n"
        "4) current_role_start_info: The text snippet indicating when they started their current executive/director role (e.g., 'since 2014', 'appointed in 2012'). If the answer does not state this, return null.\n"
        "5) joined_career_services_info: The text snippet indicating when they joined the current institution's career services office (e.g., 'joined in 1996', 'with the office since 1995'). If not stated, return null.\n"
        "6) masters_degree_title: The exact master's degree title text (e.g., 'Master of Science in Counseling'). If missing, return null.\n"
        "7) masters_field: The field for the master's degree (e.g., 'Counseling'). If missing, return null.\n"
        "8) masters_institution: The institution awarding the master's degree (e.g., 'Shippensburg University'). If missing, return null.\n"
        "9) edd_degree_title: The exact EdD degree title text (e.g., 'Ed.D. in Higher Education Administration'). If missing, return null.\n"
        "10) edd_field: The EdD field (e.g., 'Higher Education Administration'). If missing, return null.\n"
        "11) edd_institution: The institution awarding the EdD (e.g., the same university where they work). If missing, return null.\n"
        "12) undergraduate_degree_title: The exact undergraduate degree title text (e.g., 'B.A. in Psychology'). If missing, return null.\n"
        "13) undergraduate_major: The undergraduate major (e.g., 'Psychology'). If missing, return null.\n"
        "14) person_profile_url: If the answer includes a URL to the person's profile/bio page, extract it. Otherwise, null.\n"
        "15) university_homepage_url: If the answer includes a URL to the university homepage or 'About' page, extract it. Otherwise, null.\n"
        "16) source_urls: Extract all other URLs cited in the answer that pertain to this person or university. Return a list of URLs. If none, return an empty list.\n"
        "Return a JSON object with these fields. Do not invent or infer anything not explicitly in the answer. For URLs, only include valid ones appearing in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def collect_sources(p: PersonExtraction) -> List[str]:
    urls: List[str] = []
    if p.person_profile_url and p.person_profile_url.strip():
        urls.append(p.person_profile_url.strip())
    if p.university_homepage_url and p.university_homepage_url.strip():
        urls.append(p.university_homepage_url.strip())
    for u in p.source_urls:
        if u and u.strip():
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def safe_name(p: PersonExtraction) -> str:
    return p.full_name or "the individual"


def safe_uni(p: PersonExtraction) -> str:
    return p.university_name or "the university"


# --------------------------------------------------------------------------- #
# Tree construction and verification                                          #
# --------------------------------------------------------------------------- #
async def add_output_information_checks(evaluator: Evaluator, root_node, person: PersonExtraction) -> None:
    output_node = evaluator.add_parallel(
        id="Output_Information",
        desc="Required output fields are provided.",
        parent=root_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(person.full_name and person.full_name.strip()),
        id="Full_Name_Provided",
        desc="The person's full name is provided.",
        parent=output_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(person.university_name and person.university_name.strip()),
        id="University_Name_Provided",
        desc="The name of the university where the person works is provided.",
        parent=output_node,
        critical=True
    )


async def add_eligibility_checks(evaluator: Evaluator, root_node, person: PersonExtraction) -> None:
    elig_node = evaluator.add_parallel(
        id="Eligibility_Criteria",
        desc="All professional, location, tenure, and educational criteria from the question are satisfied.",
        parent=root_node,
        critical=True
    )

    sources = collect_sources(person)

    # 1) Pennsylvania university
    pa_uni_leaf = evaluator.add_leaf(
        id="Pennsylvania_University",
        desc="The individual works at a university located in Pennsylvania.",
        parent=elig_node,
        critical=True
    )
    claim_pa = f"The university '{safe_uni(person)}' is located in Pennsylvania."
    await evaluator.verify(
        claim=claim_pa,
        node=pa_uni_leaf,
        sources=sources,
        additional_instruction=(
            "Verify the university's location. Accept clearly stated location in Pennsylvania (e.g., city/state on About page or Wikipedia). "
            "If the page indicates the main campus is in Pennsylvania, consider this Correct even if the university has campuses elsewhere. "
            "If location is unclear or outside Pennsylvania, mark Incorrect."
        ),
    )

    # 2) Director-level in career services
    dir_leaf = evaluator.add_leaf(
        id="Career_Services_Director_Level",
        desc="The individual holds an executive director or director-level position in university career services.",
        parent=elig_node,
        critical=True
    )
    claim_dir = (
        f"{safe_name(person)} holds a director-level role (e.g., Director, Executive Director, Senior Director) "
        f"in the career services office at {safe_uni(person)}. Their role title is '{person.role_title or ''}'."
    )
    await evaluator.verify(
        claim=claim_dir,
        node=dir_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the role is within the university's career services function and is director-level (Director, Executive Director, Senior Director). "
            "Titles like 'Assistant Director' do NOT qualify. If the page clearly shows Director-level within Career Services, mark Correct; otherwise Incorrect."
        ),
    )

    # 3) In current role since 2018 or earlier
    since2018_leaf = evaluator.add_leaf(
        id="In_Current_Role_Since_2018_Or_Earlier",
        desc="The individual has been serving in their current executive/director role since 2018 or earlier.",
        parent=elig_node,
        critical=True
    )
    claim_since2018 = (
        f"{safe_name(person)} has been serving in their current executive/director role since 2018 or earlier at {safe_uni(person)}."
    )
    await evaluator.verify(
        claim=claim_since2018,
        node=since2018_leaf,
        sources=sources,
        additional_instruction=(
            "Look for a start year of the current executive/director role. Pass only if the year is 2018 or earlier "
            "(e.g., 'appointed in 2012', 'serving since 2010'). If the start year is 2019 or later, or no year is stated, mark Incorrect."
        ),
    )

    # 4) Joined career services by 1998 or earlier
    joined1998_leaf = evaluator.add_leaf(
        id="Joined_Career_Services_By_1998_Or_Earlier",
        desc="The individual joined their current institution's career services office in 1998 or earlier.",
        parent=elig_node,
        critical=True
    )
    claim_joined1998 = (
        f"{safe_name(person)} joined the {safe_uni(person)} career services office in 1998 or earlier."
    )
    await evaluator.verify(
        claim=claim_joined1998,
        node=joined1998_leaf,
        sources=sources,
        additional_instruction=(
            "Find when the person joined the current institution's career services office. Pass only if the year is 1998 or earlier "
            "(e.g., 'joined in 1996', 'with the office since 1995'). If the year is 1999 or later, or no year is stated, mark Incorrect."
        ),
    )

    # 5) Master's in Counseling from Shippensburg University
    masters_leaf = evaluator.add_leaf(
        id="Masters_Counseling_From_Shippensburg",
        desc="The individual holds a master's degree in Counseling from Shippensburg University.",
        parent=elig_node,
        critical=True
    )
    claim_masters = (
        f"{safe_name(person)} holds a master's degree in Counseling from Shippensburg University."
    )
    await evaluator.verify(
        claim=claim_masters,
        node=masters_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm the master's degree is specifically in Counseling and awarded by Shippensburg University. "
            "Accept common variants like 'Master of Science in Counseling' or 'M.S. in Counseling'."
        ),
    )

    # 6) EdD in Higher Education Administration from employer university
    edd_leaf = evaluator.add_leaf(
        id="EdD_HigherEdAdmin_From_Employer_University",
        desc="The individual holds an EdD in Higher Education Administration from the same university where they currently work.",
        parent=elig_node,
        critical=True
    )
    claim_edd = (
        f"{safe_name(person)} holds an EdD in Higher Education Administration from {safe_uni(person)}."
    )
    await evaluator.verify(
        claim=claim_edd,
        node=edd_leaf,
        sources=sources,
        additional_instruction=(
            "Verify that the EdD (Ed.D.) field is Higher Education Administration (or a very close variant such as Higher Education Leadership) "
            "and that the awarding institution is the same university where the person currently works."
        ),
    )

    # 7) Undergraduate major in Psychology
    ug_leaf = evaluator.add_leaf(
        id="Undergraduate_Major_Psychology",
        desc="The individual completed an undergraduate degree with a major in Psychology.",
        parent=elig_node,
        critical=True
    )
    claim_ug = (
        f"{safe_name(person)} completed an undergraduate degree with a major in Psychology."
    )
    await evaluator.verify(
        claim=claim_ug,
        node=ug_leaf,
        sources=sources,
        additional_instruction=(
            "Accept equivalents like 'B.A. in Psychology' or 'B.S. in Psychology'. If the page explicitly states Psychology as the undergraduate major, mark Correct."
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

    # Extract person information from the answer
    person = await evaluator.extract(
        prompt=prompt_extract_person_info(),
        template_class=PersonExtraction,
        extraction_name="person_info",
    )

    # Add output info checks
    await add_output_information_checks(evaluator, root, person)

    # Add eligibility criteria checks
    await add_eligibility_checks(evaluator, root, person)

    # Record extracted structured info as custom info for transparency
    evaluator.add_custom_info(
        info=person.dict(),
        info_type="extraction_snapshot",
        info_name="extracted_person_info",
    )

    # Return the evaluation summary
    return evaluator.get_summary()