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
TASK_ID = "jhu_ep_mscs_prereqs"
TASK_DESCRIPTION = """
I'm planning to apply to the Master of Science in Computer Science program at Johns Hopkins Engineering for Professionals. Before I can apply, I need to complete several prerequisite courses. Please help me by providing the following information: (1) What are the specific prerequisite courses required for admission to this program? Please provide the official university webpage URL that documents these requirements. (2) What is the minimum grade requirement that I must achieve in these prerequisite courses for them to be acceptable for admission? (3) Identify at least two accredited online platforms or institutions where I can complete these prerequisite courses. For each platform, provide a reference URL and confirm that the platform is from an accredited institution or recognized educational provider.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PrereqExtraction(BaseModel):
    """Prerequisite requirements and official source as stated in the answer."""
    official_url: Optional[str] = None
    course_list: List[str] = Field(default_factory=list)
    grade_requirement: Optional[str] = None
    grade_source_urls: List[str] = Field(default_factory=list)


class PlatformInfo(BaseModel):
    """One online platform/institution option for taking prerequisite coursework."""
    name: Optional[str] = None
    url: Optional[str] = None
    accreditation_statement: Optional[str] = None
    accreditation_urls: List[str] = Field(default_factory=list)
    prereq_courses_statement: Optional[str] = None
    prereq_courses_urls: List[str] = Field(default_factory=list)


class PlatformsExtraction(BaseModel):
    """All platforms extracted from the answer."""
    platforms: List[PlatformInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_prereq_info() -> str:
    return """
    Extract what the answer states about the prerequisite requirements for the Johns Hopkins Engineering for Professionals (EP) Master of Science in Computer Science program.

    Return the following fields:
    - official_url: The URL (must be explicitly present in the answer text) to an official Johns Hopkins EP/University webpage that documents program prerequisites (e.g., admission requirements or prerequisite coursework). Return null if not present. Do not invent URLs.
    - course_list: A list of prerequisite subject areas or specific course titles mentioned in the answer (e.g., "Data Structures", "Discrete Mathematics", "Algorithms", "Linear Algebra", "Calculus", "Programming in C++/Java"). If none are mentioned, return an empty array.
    - grade_requirement: The minimum acceptable grade (e.g., "B or higher", "3.0 or better") the answer claims is required for prerequisite courses to be acceptable. Return null if not specified.
    - grade_source_urls: Any URLs in the answer that are cited specifically for the grade requirement statement (can be the same official URL or a separate official page). If none are mentioned for grade, return an empty array.

    IMPORTANT:
    - Only extract URLs that are explicitly present in the provided answer text (plain URLs or markdown links).
    - Do not infer or invent information not present in the answer.
    """


def prompt_extract_platforms() -> str:
    return """
    Extract at least two online platforms or institutions mentioned in the answer where prerequisite-type courses can be completed.

    For each platform, return an object with:
    - name: The platform/institution name (string).
    - url: A reference URL provided in the answer for this platform (string). Return null if not present. Do not invent URLs.
    - accreditation_statement: The answer’s statement or claim text (verbatim or summarized) that indicates this platform/institution is accredited or is a recognized educational provider. If none provided, return null.
    - accreditation_urls: Any URLs provided in the answer supporting accreditation or recognized-provider claims for this platform (e.g., the platform’s accreditation page or an accreditor listing).
    - prereq_courses_statement: The answer’s statement or claim (verbatim or summarized) that the platform offers prerequisite-relevant coursework (e.g., programming, data structures, discrete mathematics).
    - prereq_courses_urls: Any URLs in the answer that point to specific course catalogs or relevant course pages for prerequisites.

    Extract all such platforms mentioned. If fewer than two are present, return what is available.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _stringify_course_list(courses: List[str]) -> str:
    if not courses:
        return ""
    return "; ".join([c.strip() for c in courses if c and c.strip()])


def _combine_unique_urls(urls: List[Optional[str]], extra: List[str]) -> List[str]:
    all_urls = []
    for u in urls:
        if isinstance(u, str) and u.strip():
            all_urls.append(u.strip())
    for u in extra:
        if isinstance(u, str) and u.strip():
            all_urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in all_urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _select_first_n_platforms(platforms: List[PlatformInfo], n: int = 2) -> List[PlatformInfo]:
    selected = []
    for p in platforms:
        if len(selected) >= n:
            break
        selected.append(p)
    # pad if fewer than n
    while len(selected) < n:
        selected.append(PlatformInfo())
    return selected


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_prerequisite_requirements_subtree(
    evaluator: Evaluator,
    parent_node,
    prereq: PrereqExtraction
) -> None:
    """
    Build the subtree for prerequisite requirements and verify against official JHU sources.
    """
    prereq_node = evaluator.add_sequential(
        id="prerequisite_requirements",
        desc="Identify the program’s prerequisite courses/subject areas and cite an official Johns Hopkins source documenting them.",
        parent=parent_node,
        critical=True
    )

    # 1) Official URL existence (critical)
    official_url_exists = evaluator.add_custom_node(
        result=(prereq.official_url is not None and isinstance(prereq.official_url, str) and prereq.official_url.strip() != "" and ("jhu.edu" in prereq.official_url or "johnshopkins" in prereq.official_url)),
        id="official_prereq_url_exists",
        desc="Official JHU prerequisite URL is provided and looks like a Johns Hopkins domain.",
        parent=prereq_node,
        critical=True
    )

    # 2) Official URL verifies it documents prerequisites (critical leaf)
    official_prereq_url_leaf = evaluator.add_leaf(
        id="official_prereq_url",
        desc="Provide a valid official Johns Hopkins (university/program) webpage URL that documents the prerequisite requirements.",
        parent=prereq_node,
        critical=True
    )
    official_url_claim = (
        "This URL is an official Johns Hopkins University/Engineering for Professionals webpage that includes or documents "
        "the prerequisite requirements for the MS in Computer Science (Engineering for Professionals) program."
    )
    await evaluator.verify(
        claim=official_url_claim,
        node=official_prereq_url_leaf,
        sources=prereq.official_url,
        additional_instruction="Verify that the page is on a Johns Hopkins domain (e.g., ep.jhu.edu) and contains clear prerequisite information for the EP MS in Computer Science."
    )

    # 3) Prerequisite course list existence (critical)
    prereq_list_exists = evaluator.add_custom_node(
        result=(bool(prereq.course_list) and len(prereq.course_list) > 0),
        id="prereq_course_list_exists",
        desc="Prerequisite subject areas/course list is provided.",
        parent=prereq_node,
        critical=True
    )

    # 4) Verify stated course list is supported on the official page (critical leaf)
    prereq_course_list_leaf = evaluator.add_leaf(
        id="prereq_course_list",
        desc="List the specific prerequisite courses/subject areas required for admission to the program.",
        parent=prereq_node,
        critical=True
    )
    course_list_str = _stringify_course_list(prereq.course_list)
    list_claim = (
        f"The official prerequisites for the EP MS in Computer Science include the following subject areas or courses: {course_list_str}."
    )
    await evaluator.verify(
        claim=list_claim,
        node=prereq_course_list_leaf,
        sources=prereq.official_url,
        additional_instruction="Allow reasonable naming variations (e.g., 'Data Structures' vs. 'Data Structure and Algorithms'). Confirm that the official page lists these areas."
    )


async def build_minimum_grade_requirement_leaf(
    evaluator: Evaluator,
    parent_node,
    prereq: PrereqExtraction
) -> None:
    """
    Verify the minimum acceptable grade requirement for prerequisite courses.
    """
    # Existence of grade requirement (critical)
    grade_exists = evaluator.add_custom_node(
        result=(prereq.grade_requirement is not None and isinstance(prereq.grade_requirement, str) and prereq.grade_requirement.strip() != ""),
        id="minimum_grade_provided",
        desc="Minimum acceptable grade requirement is stated in the answer.",
        parent=parent_node,
        critical=True
    )

    # Verification of grade requirement against sources (critical leaf)
    grade_leaf = evaluator.add_leaf(
        id="minimum_prereq_grade_requirement",
        desc="State the minimum acceptable grade requirement for prerequisite courses for them to be acceptable for admission.",
        parent=parent_node,
        critical=True
    )
    grade_claim = f"The minimum acceptable grade requirement for prerequisite courses is: {prereq.grade_requirement}."
    grade_sources = _combine_unique_urls([prereq.official_url], prereq.grade_source_urls)
    await evaluator.verify(
        claim=grade_claim,
        node=grade_leaf,
        sources=grade_sources if grade_sources else prereq.official_url,
        additional_instruction="Confirm that an official Johns Hopkins EP/University page states this minimum grade requirement for prerequisites. Accept equivalent phrasings (e.g., 'B or better' vs 'B or higher')."
    )


async def build_platform_subtree(
    evaluator: Evaluator,
    parent_node,
    platform: PlatformInfo,
    index: int
) -> None:
    """
    Build and verify one platform subtree (identification, URL, accreditation, and ability to complete prerequisites).
    """
    plat_node = evaluator.add_parallel(
        id=f"platform_{index+1}",
        desc=f"{'First' if index == 0 else 'Second'} online platform/institution option that can be used to complete prerequisite coursework.",
        parent=parent_node,
        critical=True
    )

    # Identification exists (critical)
    ident_exists = evaluator.add_custom_node(
        result=(platform.name is not None and isinstance(platform.name, str) and platform.name.strip() != ""),
        id=f"platform_{index+1}_identification",
        desc=f"Clearly identify the {'first' if index == 0 else 'second'} platform/institution (name).",
        parent=plat_node,
        critical=True
    )

    # URL existence (critical)
    url_exists = evaluator.add_custom_node(
        result=(platform.url is not None and isinstance(platform.url, str) and platform.url.strip() != ""),
        id=f"platform_{index+1}_url_exists",
        desc=f"Reference URL is provided for the {'first' if index == 0 else 'second'} platform/institution.",
        parent=plat_node,
        critical=True
    )

    # URL corresponds to platform (critical leaf)
    url_leaf = evaluator.add_leaf(
        id=f"platform_{index+1}_url",
        desc=f"Provide a reference URL for the {'first' if index == 0 else 'second'} platform/institution.",
        parent=plat_node,
        critical=True
    )
    url_claim = f"This URL is the official website or official page of the platform/institution named '{platform.name}'."
    await evaluator.verify(
        claim=url_claim,
        node=url_leaf,
        sources=platform.url,
        additional_instruction="Check that the page represents the named platform/institution (e.g., the site's branding or about page matches the name)."
    )

    # Accreditation confirmation (critical leaf)
    accred_leaf = evaluator.add_leaf(
        id=f"platform_{index+1}_accreditation_confirmation",
        desc=f"Explicitly confirm the {'first' if index == 0 else 'second'} platform/institution is an accredited institution or recognized educational provider.",
        parent=plat_node,
        critical=True
    )
    accred_claim = (
        f"The platform/institution '{platform.name}' is an accredited institution or a recognized educational provider."
    )
    accred_sources = _combine_unique_urls([platform.url], platform.accreditation_urls)
    await evaluator.verify(
        claim=accred_claim,
        node=accred_leaf,
        sources=accred_sources if accred_sources else platform.url,
        additional_instruction="Look for explicit accreditation statements (e.g., regional accreditation) or credible recognition; for non-institution platforms, confirm recognized-provider status from credible sources."
    )

    # Can complete prerequisites (critical leaf)
    can_complete_leaf = evaluator.add_leaf(
        id=f"platform_{index+1}_can_complete_prereqs",
        desc=f"Indicate that prerequisite-relevant courses can be completed via this platform/institution (i.e., it offers prerequisite-type coursework).",
        parent=plat_node,
        critical=True
    )
    prereq_claim = (
        f"The platform/institution '{platform.name}' offers courses suitable as prerequisite-type coursework for an MSCS program "
        f"(e.g., programming, data structures, discrete mathematics, calculus, linear algebra, algorithms)."
    )
    prereq_sources = _combine_unique_urls([platform.url], platform.prereq_courses_urls)
    await evaluator.verify(
        claim=prereq_claim,
        node=can_complete_leaf,
        sources=prereq_sources if prereq_sources else platform.url,
        additional_instruction="Confirm the presence of relevant computer science/math foundational courses or equivalent prerequisite-level offerings."
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
    Evaluate an answer for JHU EP MSCS prerequisites task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates main sections in parallel
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

    # Extract prerequisite info and platform options
    prereq_info, platforms_info = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_prereq_info(),
            template_class=PrereqExtraction,
            extraction_name="prereq_info"
        ),
        evaluator.extract(
            prompt=prompt_extract_platforms(),
            template_class=PlatformsExtraction,
            extraction_name="platforms_info"
        )
    )

    # Create a critical main node under root to represent overall task (since root is non-critical by framework)
    main_node = evaluator.add_parallel(
        id="task_main",
        desc="Provide (1) the program prerequisite courses with an official JHU URL, (2) the minimum acceptable grade for prerequisites, and (3) at least two accredited/recognized online options to complete the prerequisites with URLs and accreditation confirmation.",
        parent=root,
        critical=True
    )

    # 1) Prerequisite requirements subtree
    await build_prerequisite_requirements_subtree(evaluator, main_node, prereq_info)

    # 2) Minimum grade requirement leaf (with existence gating)
    await build_minimum_grade_requirement_leaf(evaluator, main_node, prereq_info)

    # 3) Online platform options subtree (need at least two options)
    online_opts_node = evaluator.add_parallel(
        id="online_prereq_options",
        desc="Provide at least two online platforms/institutions where the prerequisite courses can be completed; for each, include a reference URL and accreditation/recognized-provider confirmation.",
        parent=main_node,
        critical=True
    )

    selected_platforms = _select_first_n_platforms(platforms_info.platforms, n=2)
    for idx, plat in enumerate(selected_platforms):
        await build_platform_subtree(evaluator, online_opts_node, plat, idx)

    # Add custom info for transparency
    evaluator.add_custom_info(
        info={
            "extracted_platform_count": len(platforms_info.platforms),
            "used_platform_count": len(selected_platforms)
        },
        info_type="extraction_stats",
        info_name="platforms_count_info"
    )

    # Return final structured evaluation summary
    return evaluator.get_summary()