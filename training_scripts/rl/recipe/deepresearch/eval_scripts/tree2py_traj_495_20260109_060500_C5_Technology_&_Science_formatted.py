import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "cmu_ri_career_2025_safety_hri"
TASK_DESCRIPTION = (
    "Who is the assistant professor at the Robotics Institute of Carnegie Mellon University who received an NSF CAREER "
    "award in 2025 for a 5-year project, completed their PhD from the University of California, Berkeley in 2022, "
    "earned their Bachelor's degree between 2015 and 2017 (inclusive), and whose research focuses specifically on robot "
    "safety and safe human-robot interaction?"
)


class ProfessorExtraction(BaseModel):
    # Identity
    name: Optional[str] = None

    # Position and affiliation
    title: Optional[str] = None
    affiliation: Optional[str] = None

    # NSF CAREER
    nsf_career_year: Optional[str] = None
    nsf_career_duration_years: Optional[str] = None

    # Education
    phd_institution: Optional[str] = None
    phd_year: Optional[str] = None
    bachelors_institution: Optional[str] = None
    bachelors_year: Optional[str] = None

    # Research focus
    research_focus: Optional[str] = None

    # Source URLs grouped by aspect
    position_sources: List[str] = Field(default_factory=list)
    affiliation_sources: List[str] = Field(default_factory=list)
    nsf_sources: List[str] = Field(default_factory=list)
    phd_sources: List[str] = Field(default_factory=list)
    bachelors_sources: List[str] = Field(default_factory=list)
    research_sources: List[str] = Field(default_factory=list)
    general_sources: List[str] = Field(default_factory=list)


def prompt_extract_professor() -> str:
    return (
        "Extract the single professor identified in the answer who is claimed to meet all of the constraints. "
        "Return the following fields strictly based on what is explicitly written in the answer. "
        "Do not invent any information. If a field is missing, return null, and if sources are missing, return an empty list.\n\n"
        "Required fields:\n"
        "1) name: The full name of the professor.\n"
        "2) title: The academic title stated (e.g., Assistant Professor).\n"
        "3) affiliation: The stated institutional affiliation or unit (e.g., 'Robotics Institute, Carnegie Mellon University').\n"
        "4) nsf_career_year: The year of the NSF CAREER award, if specified.\n"
        "5) nsf_career_duration_years: The duration string for the CAREER project, if specified (e.g., '5 years').\n"
        "6) phd_institution: Institution of the PhD.\n"
        "7) phd_year: Year the PhD was completed.\n"
        "8) bachelors_institution: Institution for the Bachelor's degree, if stated.\n"
        "9) bachelors_year: Year the Bachelor's degree was earned, if stated.\n"
        "10) research_focus: The stated research focus text (short summary or keywords) as given in the answer.\n"
        "\n"
        "Also extract URLs explicitly cited in the answer grouped by aspect. Only include actual URLs (plain or from markdown links):\n"
        "- position_sources: URLs that support the person's title (Assistant Professor) or role.\n"
        "- affiliation_sources: URLs that support affiliation with the Robotics Institute at Carnegie Mellon University.\n"
        "- nsf_sources: URLs that support the NSF CAREER award details (including year and duration), such as NSF award pages or official announcements.\n"
        "- phd_sources: URLs that support the PhD information (institution and year).\n"
        "- bachelors_sources: URLs that support the Bachelor's degree year.\n"
        "- research_sources: URLs that support the research focus on robot safety and safe human-robot interaction.\n"
        "- general_sources: Any other URLs cited about this person not captured above.\n"
        "\n"
        "Return a single JSON object exactly matching the schema."
    )


def _combine_sources(*lists: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in lists:
        for url in lst or []:
            if url and url not in seen:
                seen.add(url)
                result.append(url)
    return result


def _parse_year(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    s = value.strip()
    # Try to find a 4-digit year within the string
    import re
    m = re.search(r"\b(19|20)\d{2}\b", s)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    # If string is a plain 4-digit number
    if s.isdigit() and len(s) == 4:
        try:
            return int(s)
        except Exception:
            return None
    return None


async def verify_position_and_affiliation(evaluator: Evaluator, parent_node, info: ProfessorExtraction) -> None:
    pos_node = evaluator.add_parallel(
        id="Position_and_Affiliation",
        desc="Verify the professor's title and institutional affiliation as specified",
        parent=parent_node,
        critical=True
    )

    # Evidence existence checks (critical gating)
    evaluator.add_custom_node(
        result=len(info.position_sources) > 0 or len(info.affiliation_sources) > 0 or len(info.general_sources) > 0,
        id="Position_Affiliation_Evidence_Provided",
        desc="Position/affiliation evidence URLs are provided in the answer",
        parent=pos_node,
        critical=True
    )

    # Assistant Professor title
    title_leaf = evaluator.add_leaf(
        id="Assistant_Professor_Title",
        desc="The professor holds the title of Assistant Professor",
        parent=pos_node,
        critical=True
    )
    title_sources = _combine_sources(info.position_sources, info.affiliation_sources, info.general_sources)
    title_claim = f"The person '{info.name or 'the professor'}' holds the title 'Assistant Professor'."
    await evaluator.verify(
        claim=title_claim,
        node=title_leaf,
        sources=title_sources,
        additional_instruction=(
            "Verify that the cited page(s) explicitly show the academic title 'Assistant Professor' for this person. "
            "Minor variations like 'Asst. Professor' or inclusion of department/unit alongside the title should be accepted."
        )
    )

    # Robotics Institute at CMU affiliation (as of 2025)
    aff_leaf = evaluator.add_leaf(
        id="RI_at_CMU_Affiliation_As_of_2025",
        desc="The professor is affiliated with the Robotics Institute at Carnegie Mellon University as of 2025",
        parent=pos_node,
        critical=True
    )
    aff_sources = _combine_sources(info.affiliation_sources, info.position_sources, info.general_sources)
    aff_claim = (
        f"As of 2025, the person '{info.name or 'the professor'}' is affiliated with the Robotics Institute at Carnegie Mellon University."
    )
    await evaluator.verify(
        claim=aff_claim,
        node=aff_leaf,
        sources=aff_sources,
        additional_instruction=(
            "Confirm that the page(s) indicate an affiliation with the Robotics Institute at Carnegie Mellon University. "
            "If the page is a current official profile or lab page stating the affiliation, it suffices to consider it valid for 'as of 2025' even if the date is not explicitly printed."
        )
    )


async def verify_nsf_career(evaluator: Evaluator, parent_node, info: ProfessorExtraction) -> None:
    nsf_node = evaluator.add_parallel(
        id="NSF_CAREER_Award",
        desc="Verify NSF CAREER award requirements",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(info.nsf_sources) > 0,
        id="NSF_Evidence_Provided",
        desc="NSF CAREER evidence URLs are provided in the answer",
        parent=nsf_node,
        critical=True
    )

    # CAREER in 2025
    career_year_leaf = evaluator.add_leaf(
        id="NSF_CAREER_in_2025",
        desc="The professor received an NSF CAREER award in 2025",
        parent=nsf_node,
        critical=True
    )
    career_year_claim = f"In 2025, '{info.name or 'the professor'}' received an NSF CAREER (Faculty Early Career Development Program) award."
    await evaluator.verify(
        claim=career_year_claim,
        node=career_year_leaf,
        sources=info.nsf_sources,
        additional_instruction=(
            "Look for explicit mention of an NSF CAREER award and confirm the year 2025 on the provided page(s), "
            "such as an NSF award notice or an official institutional/news announcement."
        )
    )

    # CAREER 5-year duration
    duration_leaf = evaluator.add_leaf(
        id="CAREER_Five_Year_Duration",
        desc="The NSF CAREER award project duration is 5 years",
        parent=nsf_node,
        critical=True
    )
    duration_claim = (
        f"The NSF CAREER award for '{info.name or 'the professor'}' supports a project with a duration of five years."
    )
    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        sources=info.nsf_sources,
        additional_instruction=(
            "Confirm the project duration is five years. The page may state '5-year award', 'five years', or show start/end dates implying five years."
        )
    )


async def verify_education(evaluator: Evaluator, parent_node, info: ProfessorExtraction) -> None:
    edu_node = evaluator.add_parallel(
        id="Education",
        desc="Verify educational background requirements",
        parent=parent_node,
        critical=True
    )

    # PhD evidence
    evaluator.add_custom_node(
        result=len(info.phd_sources) > 0,
        id="PhD_Evidence_Provided",
        desc="PhD evidence URLs are provided in the answer",
        parent=edu_node,
        critical=True
    )

    # PhD UC Berkeley 2022
    phd_leaf = evaluator.add_leaf(
        id="PhD_UC_Berkeley_2022",
        desc="The professor completed their PhD from the University of California, Berkeley in 2022",
        parent=edu_node,
        critical=True
    )
    phd_claim = f"In 2022, '{info.name or 'the professor'}' completed a PhD at the University of California, Berkeley."
    await evaluator.verify(
        claim=phd_claim,
        node=phd_leaf,
        sources=info.phd_sources,
        additional_instruction=(
            "Confirm that the provided page(s) explicitly state a PhD from UC Berkeley and the year 2022 for this person."
        )
    )

    # Bachelor's evidence
    evaluator.add_custom_node(
        result=len(info.bachelors_sources) > 0,
        id="Bachelors_Evidence_Provided",
        desc="Bachelor's degree evidence URLs are provided in the answer",
        parent=edu_node,
        critical=True
    )

    # Bachelor's year supported by sources
    bachelors_year_leaf = evaluator.add_leaf(
        id="Bachelors_Year_Supported",
        desc="The professor's Bachelor's year is supported by cited sources",
        parent=edu_node,
        critical=True
    )
    if info.bachelors_year:
        bachelors_year_claim = (
            f"In {info.bachelors_year}, '{info.name or 'the professor'}' earned a Bachelor's degree."
        )
    else:
        bachelors_year_claim = (
            f"'{info.name or 'the professor'}' earned a Bachelor's degree, and the year is explicitly stated on the provided page(s)."
        )
    await evaluator.verify(
        claim=bachelors_year_claim,
        node=bachelors_year_leaf,
        sources=info.bachelors_sources,
        additional_instruction=(
            "Confirm the Bachelor's degree year for this person on the provided page(s). If the answer names a specific year, "
            "verify that the page matches it."
        )
    )

    # Bachelor's year in range [2015, 2017] inclusive (custom range check)
    b_year = _parse_year(info.bachelors_year)
    b_in_range = b_year is not None and 2015 <= b_year <= 2017
    evaluator.add_custom_node(
        result=b_in_range,
        id="Bachelors_2015_to_2017",
        desc="The professor earned their Bachelor's degree between 2015 and 2017 inclusive",
        parent=edu_node,
        critical=True
    )


async def verify_research_focus(evaluator: Evaluator, parent_node, info: ProfessorExtraction) -> None:
    res_node = evaluator.add_parallel(
        id="Research_Focus",
        desc="Verify the professor's research focus requirement",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(info.research_sources) > 0 or len(info.general_sources) > 0,
        id="Research_Evidence_Provided",
        desc="Research focus evidence URLs are provided in the answer",
        parent=res_node,
        critical=True
    )

    focus_leaf = evaluator.add_leaf(
        id="Robot_Safety_and_Safe_HRI",
        desc="The professor's research focuses specifically on robot safety and safe human-robot interaction",
        parent=res_node,
        critical=True
    )
    focus_sources = _combine_sources(info.research_sources, info.general_sources)
    focus_claim = (
        f"The research of '{info.name or 'the professor'}' focuses specifically on robot safety and safe human-robot interaction."
    )
    await evaluator.verify(
        claim=focus_claim,
        node=focus_leaf,
        sources=focus_sources,
        additional_instruction=(
            "Confirm that the page(s) explicitly indicate research emphasis on robot safety and safe human-robot interaction. "
            "Reasonable synonyms like 'safety in robotics', 'safe HRI', or 'safe human–robot interaction' should be accepted."
        )
    )


async def build_verification_tree(evaluator: Evaluator, info: ProfessorExtraction) -> None:
    # Critical overall identification node
    prof_node = evaluator.add_parallel(
        id="Professor_Identification",
        desc="Identify a professor who meets all specified criteria in the constraints",
        parent=evaluator.root,
        critical=True
    )

    # Minimal identity and evidence existence checks at top level (critical gating)
    evaluator.add_custom_node(
        result=bool(info.name and info.name.strip()),
        id="Professor_Name_Provided",
        desc="Professor's name is provided in the answer",
        parent=prof_node,
        critical=True
    )
    any_sources = len(_combine_sources(
        info.position_sources,
        info.affiliation_sources,
        info.nsf_sources,
        info.phd_sources,
        info.bachelors_sources,
        info.research_sources,
        info.general_sources
    )) > 0
    evaluator.add_custom_node(
        result=any_sources,
        id="Any_Source_Provided",
        desc="At least one source URL is provided in the answer",
        parent=prof_node,
        critical=True
    )

    # Subgroups (all critical)
    await verify_position_and_affiliation(evaluator, prof_node, info)
    await verify_nsf_career(evaluator, prof_node, info)
    await verify_education(evaluator, prof_node, info)
    await verify_research_focus(evaluator, prof_node, info)


async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    evaluator = Evaluator()
    evaluator.initialize(
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_professor(),
        template_class=ProfessorExtraction,
        extraction_name="professor_extraction"
    )

    await build_verification_tree(evaluator, extracted)

    return evaluator.get_summary()