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
TASK_ID = "cs_ms_four_universities"
TASK_DESCRIPTION = """
Identify four universities in the United States that offer a Master of Science (MS) in Computer Science program meeting all of the following criteria:

1. The program must offer both thesis and non-thesis degree options for the MS in Computer Science
2. The minimum GPA requirement for admission must be 3.0 or lower (on a 4.0 scale)
3. The Computer Science department must have at least one faculty member whose research specialization includes Artificial Intelligence or Machine Learning
4. The university must offer Teaching Assistantship (TA) positions for graduate students in Computer Science

For each university, provide:
- The full official name of the university
- A link to the official graduate program page showing the thesis and non-thesis options
- A link to the official admissions requirements page showing the minimum GPA requirement
- The name of at least one faculty member specializing in AI/ML and a link to their faculty profile page
- A link to information about Teaching Assistantship availability for Computer Science graduate students
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    university_name: Optional[str] = None
    program_page_url: Optional[str] = None
    admissions_page_url: Optional[str] = None
    faculty_name: Optional[str] = None
    faculty_profile_url: Optional[str] = None
    ta_info_url: Optional[str] = None


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to four universities mentioned in the answer that the author claims meet the specified criteria for an MS in Computer Science. For each university, extract the following fields exactly as provided:

    - university_name: The full official name of the university.
    - program_page_url: The URL to the official graduate program page that shows the thesis and non-thesis options for the MS in Computer Science. Prefer department/college/university official pages over third-party sites.
    - admissions_page_url: The URL to the official admissions requirements page that states the minimum GPA requirement (on a 4.0 scale).
    - faculty_name: The name of at least one faculty member whose specialization includes Artificial Intelligence (AI) or Machine Learning (ML). If multiple are mentioned, pick one.
    - faculty_profile_url: The URL to that faculty member's official profile page.
    - ta_info_url: The URL to information about Teaching Assistantship (TA) availability for graduate students in Computer Science. Prefer department-specific pages if available; otherwise use official graduate school/college pages.

    RULES:
    - Only extract URLs explicitly present in the answer (plain URLs or markdown links). If a URL is missing, set the field to null.
    - If the answer lists more than four universities, keep only the first four and ignore the rest.
    - Do not invent or infer any information not present in the answer.
    - If any field is missing for a university, set it to null.

    Return a JSON object with a single key 'universities', which is an array of objects, each containing exactly these six fields.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    return mapping.get(n, f"#{n}")


def _non_empty(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    index: int,
) -> None:
    """
    Build and verify the tree for a single university.
    """
    ord_word = ordinal(index + 1)

    # University-level node (non-critical to allow partial credit across universities)
    uni_node = evaluator.add_parallel(
        id=f"university_{index + 1}",
        desc=f"{ord_word} university meets all requirements with complete evidence",
        parent=parent_node,
        critical=False
    )

    # 1) University name (critical existence)
    evaluator.add_custom_node(
        result=_non_empty(uni.university_name),
        id=f"university_{index + 1}_name",
        desc="The full official name of the university is provided",
        parent=uni_node,
        critical=True
    )

    # 2) Program options (critical group)
    program_node = evaluator.add_parallel(
        id=f"university_{index + 1}_program_options",
        desc="MS in Computer Science offers both thesis and non-thesis options with evidence",
        parent=uni_node,
        critical=True
    )

    # 2.1) Program page URL existence (critical)
    evaluator.add_custom_node(
        result=_non_empty(uni.program_page_url),
        id=f"university_{index + 1}_program_page_url",
        desc="A link to the official graduate program page showing the thesis and non-thesis options is provided",
        parent=program_node,
        critical=True
    )

    # 2.2) Verify the program offers both thesis and non-thesis options (critical, by URL)
    offers_node = evaluator.add_leaf(
        id=f"university_{index + 1}_program_offers_both_options",
        desc="The university offers MS in Computer Science with both thesis and non-thesis options",
        parent=program_node,
        critical=True
    )
    await evaluator.verify(
        claim="The MS in Computer Science program offers both thesis and non-thesis (coursework/project/capstone) options.",
        node=offers_node,
        sources=uni.program_page_url,
        additional_instruction=(
            "Confirm the page is about the MS in Computer Science program at this university. "
            "Accept synonyms such as 'thesis track/option (Plan A)' and 'non-thesis track/option (Plan B)', "
            "'coursework-only', 'project option', or 'capstone' as non-thesis. "
            "Reject pages that only mention one option or that refer to a different degree."
        ),
    )

    # 3) GPA requirement (critical group)
    gpa_node = evaluator.add_parallel(
        id=f"university_{index + 1}_gpa_requirement",
        desc="Minimum GPA requirement is 3.0 or lower with evidence",
        parent=uni_node,
        critical=True
    )

    # 3.1) Admissions page URL existence (critical)
    evaluator.add_custom_node(
        result=_non_empty(uni.admissions_page_url),
        id=f"university_{index + 1}_admissions_page_url",
        desc="A link to the official admissions requirements page showing the minimum GPA requirement is provided",
        parent=gpa_node,
        critical=True
    )

    # 3.2) Verify min GPA <= 3.0 (critical, by URL)
    gpa_leaf = evaluator.add_leaf(
        id=f"university_{index + 1}_gpa_threshold_met",
        desc="The minimum GPA requirement for admission is 3.0 or lower on a 4.0 scale",
        parent=gpa_node,
        critical=True
    )
    await evaluator.verify(
        claim="The minimum GPA requirement for admission to the MS in Computer Science is 3.0 or lower (on a 4.0 scale).",
        node=gpa_leaf,
        sources=uni.admissions_page_url,
        additional_instruction=(
            "Treat the claim as supported only if the page explicitly states a minimum GPA requirement "
            "that is ≤ 3.0 (e.g., 3.0, 2.75, 2.8). "
            "If the page says 'no minimum', 'holistic review' without a number, or gives a minimum > 3.0, "
            "the claim is NOT supported. "
            "If multiple minima are listed (grad school vs department), use the one applicable to MS CS."
        ),
    )

    # 4) AI/ML faculty (critical group)
    faculty_node = evaluator.add_parallel(
        id=f"university_{index + 1}_ai_faculty",
        desc="At least one AI/ML faculty member is identified with evidence",
        parent=uni_node,
        critical=True
    )

    # 4.1) Faculty name existence (critical)
    evaluator.add_custom_node(
        result=_non_empty(uni.faculty_name),
        id=f"university_{index + 1}_faculty_name",
        desc="The name of at least one faculty member specializing in AI/ML is provided",
        parent=faculty_node,
        critical=True
    )

    # 4.2) Faculty profile URL existence (critical)
    evaluator.add_custom_node(
        result=_non_empty(uni.faculty_profile_url),
        id=f"university_{index + 1}_faculty_profile_url",
        desc="A link to the faculty member's profile page is provided",
        parent=faculty_node,
        critical=True
    )

    # 4.3) Verify AI/ML specialization (critical, by URL)
    fac_spec_leaf = evaluator.add_leaf(
        id=f"university_{index + 1}_faculty_specialization",
        desc="At least one faculty member specializes in Artificial Intelligence or Machine Learning",
        parent=faculty_node,
        critical=True
    )
    fac_name_for_claim = uni.faculty_name or "the faculty member"
    await evaluator.verify(
        claim=f"{fac_name_for_claim} specializes in Artificial Intelligence or Machine Learning.",
        node=fac_spec_leaf,
        sources=uni.faculty_profile_url,
        additional_instruction=(
            "From the official faculty profile page, confirm the research area explicitly includes AI/ML "
            "or direct synonyms: 'Artificial Intelligence', 'Machine Learning', 'Deep Learning', "
            "'Neural Networks', 'Reinforcement Learning'. "
            "General 'Data Science' alone does not qualify unless it explicitly includes ML/AI."
        ),
    )

    # 5) TA availability (critical group)
    ta_node = evaluator.add_parallel(
        id=f"university_{index + 1}_ta_availability",
        desc="Teaching Assistantship availability is confirmed with evidence",
        parent=uni_node,
        critical=True
    )

    # 5.1) TA info URL existence (critical)
    evaluator.add_custom_node(
        result=_non_empty(uni.ta_info_url),
        id=f"university_{index + 1}_ta_info_url",
        desc="A link to information about Teaching Assistantship availability is provided",
        parent=ta_node,
        critical=True
    )

    # 5.2) Verify TA positions available (critical, by URL)
    ta_leaf = evaluator.add_leaf(
        id=f"university_{index + 1}_ta_positions_available",
        desc="Teaching Assistantship positions are available for Computer Science graduate students",
        parent=ta_node,
        critical=True
    )
    await evaluator.verify(
        claim="Teaching Assistantship (TA) positions are available to graduate students in the Computer Science department.",
        node=ta_leaf,
        sources=uni.ta_info_url,
        additional_instruction=(
            "Confirm the page indicates TA positions exist and are available to CS graduate students "
            "(including MS CS). Accept CS department pages or official grad school pages that explicitly "
            "mention TAs for CS. If the page explicitly restricts TAs to PhD only and excludes MS, "
            "treat as NOT supported."
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the MS in Computer Science universities task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates universities independently
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

    # IMPORTANT: Make root non-critical to allow partial credit across universities
    root.critical = False

    # Extract universities from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Keep first four universities; pad with empty items if fewer provided
    universities = (extracted.universities or [])[:4]
    while len(universities) < 4:
        universities.append(UniversityItem())

    # Record a custom info summary
    evaluator.add_custom_info(
        info={"extracted_university_count": len(extracted.universities or []), "evaluated_count": 4},
        info_type="extraction_stats",
        info_name="extraction_statistics"
    )

    # Build verification subtrees for each university
    for idx, uni in enumerate(universities):
        await verify_university(evaluator, root, uni, idx)

    # Return evaluation summary
    return evaluator.get_summary()