import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "superintendent_position_research_4_states"
TASK_DESCRIPTION = """Research and provide specific information about the largest public school district (by student enrollment) in each of the following four U.S. states: Georgia, California, Texas, and Florida. For each state's largest district, provide the following information as of 2026:

1. The official name of the largest public school district
2. The current student enrollment number (2024-2025 or 2025-2026 school year)
3. The annual salary or salary range for the superintendent position
4. The minimum years of leadership or administrative experience typically required for the superintendent role

All information must be verifiable through official district websites, news articles, or government sources, and each piece of information must include a reference URL.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StateDistrictInfo(BaseModel):
    district_name: Optional[str] = None
    district_name_urls: List[str] = Field(default_factory=list)

    enrollment: Optional[str] = None
    enrollment_urls: List[str] = Field(default_factory=list)

    superintendent_salary: Optional[str] = None
    superintendent_salary_urls: List[str] = Field(default_factory=list)

    experience_requirement: Optional[str] = None
    experience_requirement_urls: List[str] = Field(default_factory=list)


class FourStatesExtraction(BaseModel):
    georgia: Optional[StateDistrictInfo] = None
    california: Optional[StateDistrictInfo] = None
    texas: Optional[StateDistrictInfo] = None
    florida: Optional[StateDistrictInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_states_info() -> str:
    return """
    Extract structured information for each of the four states: Georgia, California, Texas, and Florida.
    We need, for each state's largest public school district by student enrollment:
    1) district_name (string, as written in the answer)
    2) district_name_urls (array of URLs that support that the named district is the largest by enrollment and/or confirm the official district name)
    3) enrollment (string exactly as stated in the answer; may include commas, the words 'approximately', or a range)
    4) enrollment_urls (array of URLs that support the enrollment figure and indicate the school year 2024–2025 or 2025–2026)
    5) superintendent_salary (string exactly as stated in the answer; may be a single number or a range; may include currency symbols)
    6) superintendent_salary_urls (array of URLs that support the salary/compensation figure or range for the superintendent)
    7) experience_requirement (string exactly as stated; e.g., '10 years of administrative leadership experience')
    8) experience_requirement_urls (array of URLs that support the stated minimum years of leadership/administrative experience for the superintendent role)

    IMPORTANT:
    - Only extract information explicitly present in the answer.
    - For each URL field, extract all URLs mentioned for that item in the answer text (including markdown links), deduplicate them, and include only valid URLs.
    - If a value is missing in the answer, set the corresponding string field to null and return an empty array for the associated URLs.
    - If the answer bundles sources in a 'Sources' section, assign each URL to the specific items it supports as best as possible. If ambiguous, include the URL for multiple relevant items.

    Return a JSON object with top-level keys: 'georgia', 'california', 'texas', and 'florida'.
    Each key should map to an object with the eight fields defined above.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_value_and_urls(value: Optional[str], urls: Optional[List[str]]) -> bool:
    return bool(value and str(value).strip()) and bool(urls and len(urls) > 0)


def _state_title(state_key: str) -> str:
    return {
        "georgia": "Georgia",
        "california": "California",
        "texas": "Texas",
        "florida": "Florida",
    }[state_key]


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _build_name_verification(
    evaluator: Evaluator,
    parent,
    state_key: str,
    info: Optional[StateDistrictInfo]
) -> None:
    state_name = _state_title(state_key)
    group = evaluator.add_sequential(
        id=f"{state_name}_District_Name_group",
        desc=f"{state_name}: Official name of the largest district (by enrollment) – existence then verification",
        parent=parent,
        critical=True
    )

    exists = evaluator.add_custom_node(
        result=_has_value_and_urls(info.district_name if info else None, info.district_name_urls if info else []),
        id=f"{state_name}_District_Name_exists",
        desc=f"{state_name}: District name value and at least one supporting URL are provided",
        parent=group,
        critical=True
    )

    verify_leaf = evaluator.add_leaf(
        id=f"{state_name}_District_Name",
        desc=f"Correctly identify the official name of the largest public school district (by enrollment) in {state_name}",
        parent=group,
        critical=True
    )

    claim_name = info.district_name or ""
    await evaluator.verify(
        claim=(
            f"The largest public school district by student enrollment in {state_name} is named '{claim_name}'. "
            f"The provided source should clearly support that this district is the largest in {state_name} by enrollment "
            f"and/or explicitly identify the district by this official name."
        ),
        node=verify_leaf,
        sources=(info.district_name_urls if info else None),
        additional_instruction=(
            "Accept pages that explicitly state the district is the largest by enrollment in the state, "
            "or show comparative enrollment data indicating it is largest. Allow minor name variations "
            "(e.g., 'ISD' vs 'Independent School District', 'Public Schools' vs 'Public School District')."
        )
    )


async def _build_enrollment_verification(
    evaluator: Evaluator,
    parent,
    state_key: str,
    info: Optional[StateDistrictInfo]
) -> None:
    state_name = _state_title(state_key)
    group = evaluator.add_sequential(
        id=f"{state_name}_Enrollment_group",
        desc=f"{state_name}: Current student enrollment (2024–2025 or 2025–2026) – existence then verification",
        parent=parent,
        critical=True
    )

    exists = evaluator.add_custom_node(
        result=_has_value_and_urls(info.enrollment if info else None, info.enrollment_urls if info else []),
        id=f"{state_name}_Enrollment_exists",
        desc=f"{state_name}: Enrollment value and at least one supporting URL are provided",
        parent=group,
        critical=True
    )

    verify_leaf = evaluator.add_leaf(
        id=f"{state_name}_Enrollment",
        desc=f"Provide the current student enrollment number for {state_name}'s largest district (2024-2025 or 2025-2026 school year data)",
        parent=group,
        critical=True
    )

    district_name_text = info.district_name or f"the largest district in {state_name}"
    enrollment_text = info.enrollment or ""
    await evaluator.verify(
        claim=(
            f"For the 2024-2025 or 2025-2026 school year, the student enrollment of {district_name_text} "
            f"is {enrollment_text}."
        ),
        node=verify_leaf,
        sources=(info.enrollment_urls if info else None),
        additional_instruction=(
            "Confirm that the cited page(s) provide the enrollment for the 2024–2025 or 2025–2026 school year. "
            "If multiple figures are shown, prefer the explicitly labeled 2024–25 or 2025–26 number. "
            "Allow reasonable rounding and formatting differences (commas, 'approximately', etc.). "
            "If the page only provides earlier years (e.g., 2023–24 or before), consider the claim unsupported."
        )
    )


async def _build_salary_verification(
    evaluator: Evaluator,
    parent,
    state_key: str,
    info: Optional[StateDistrictInfo]
) -> None:
    state_name = _state_title(state_key)
    group = evaluator.add_sequential(
        id=f"{state_name}_Superintendent_Salary_group",
        desc=f"{state_name}: Superintendent annual salary or range – existence then verification",
        parent=parent,
        critical=True
    )

    exists = evaluator.add_custom_node(
        result=_has_value_and_urls(info.superintendent_salary if info else None, info.superintendent_salary_urls if info else []),
        id=f"{state_name}_Superintendent_Salary_exists",
        desc=f"{state_name}: Superintendent salary value and at least one supporting URL are provided",
        parent=group,
        critical=True
    )

    verify_leaf = evaluator.add_leaf(
        id=f"{state_name}_Superintendent_Salary",
        desc=f"Provide the salary range or annual compensation for the superintendent position in {state_name}'s largest district",
        parent=group,
        critical=True
    )

    district_name_text = info.district_name or f"the largest district in {state_name}"
    salary_text = info.superintendent_salary or ""
    await evaluator.verify(
        claim=(
            f"The superintendent's annual salary or salary range for {district_name_text} is {salary_text}."
        ),
        node=verify_leaf,
        sources=(info.superintendent_salary_urls if info else None),
        additional_instruction=(
            "Accept either a single annual salary figure or a range. If total compensation is stated (base + allowances), "
            "treat it as salary for this task. Allow minor rounding differences. Prefer official contracts, board documents, "
            "or credible news/government sources."
        )
    )


async def _build_experience_verification(
    evaluator: Evaluator,
    parent,
    state_key: str,
    info: Optional[StateDistrictInfo]
) -> None:
    state_name = _state_title(state_key)
    group = evaluator.add_sequential(
        id=f"{state_name}_Experience_Requirement_group",
        desc=f"{state_name}: Minimum years of leadership/administrative experience – existence then verification",
        parent=parent,
        critical=True
    )

    exists = evaluator.add_custom_node(
        result=_has_value_and_urls(info.experience_requirement if info else None, info.experience_requirement_urls if info else []),
        id=f"{state_name}_Experience_Requirement_exists",
        desc=f"{state_name}: Experience requirement value and at least one supporting URL are provided",
        parent=group,
        critical=True
    )

    verify_leaf = evaluator.add_leaf(
        id=f"{state_name}_Experience_Requirement",
        desc=f"Specify the minimum years of leadership/administrative experience typically required for superintendent positions in {state_name}'s largest district",
        parent=group,
        critical=True
    )

    district_name_text = info.district_name or f"the largest district in {state_name}"
    experience_text = info.experience_requirement or ""
    await evaluator.verify(
        claim=(
            f"The minimum years of leadership/administrative experience typically required for the superintendent role in "
            f"{district_name_text} is {experience_text}."
        ),
        node=verify_leaf,
        sources=(info.experience_requirement_urls if info else None),
        additional_instruction=(
            "Accept wording such as 'at least X years', 'minimum X years', or 'X+ years' for administrative or leadership "
            "experience relevant to superintendent qualifications. Prefer official job postings, HR policy documents, "
            "board policies, or credible government/news sources."
        )
    )


async def _verify_state_block(
    evaluator: Evaluator,
    parent,
    state_key: str,
    info: Optional[StateDistrictInfo]
) -> None:
    state_name = _state_title(state_key)

    state_node = evaluator.add_parallel(
        id=f"{state_name}_Largest_District",
        desc=f"Provide complete information about {state_name}'s largest public school district (by enrollment)",
        parent=parent,
        critical=False
    )

    await _build_name_verification(evaluator, state_node, state_key, info or StateDistrictInfo())
    await _build_enrollment_verification(evaluator, state_node, state_key, info or StateDistrictInfo())
    await _build_salary_verification(evaluator, state_node, state_key, info or StateDistrictInfo())
    await _build_experience_verification(evaluator, state_node, state_key, info or StateDistrictInfo())


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
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
    """
    Evaluate an answer for superintendent position research across GA, CA, TX, FL.
    """
    # Initialize evaluator with a parallel root as rubric requires independent state blocks
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

    # Extract all states information from the answer
    extracted: FourStatesExtraction = await evaluator.extract(
        prompt=prompt_extract_states_info(),
        template_class=FourStatesExtraction,
        extraction_name="extracted_states_info"
    )

    # Build top-level node corresponding to rubric root
    top_node = evaluator.add_parallel(
        id="Superintendent_Position_Research",
        desc="Research and provide specific information about superintendent positions and the largest public school districts in four different U.S. states",
        parent=root,
        critical=False
    )

    # Verify each state block (parallel, independent)
    await _verify_state_block(evaluator, top_node, "georgia", extracted.georgia)
    await _verify_state_block(evaluator, top_node, "california", extracted.california)
    await _verify_state_block(evaluator, top_node, "texas", extracted.texas)
    await _verify_state_block(evaluator, top_node, "florida", extracted.florida)

    # Return structured summary
    return evaluator.get_summary()