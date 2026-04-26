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
TASK_ID = "us_school_districts_ranking_demographics"
TASK_DESCRIPTION = """
In the United States public school system, identify three specific large school districts, where each district satisfies a unique combination of ranking, size, and demographic requirements. For each identified district, provide: (1) the district's official name, (2) the state in which it is located, (3) its current student enrollment, and (4) supporting reference URLs for verification.

District 1 Requirements:
The first district must simultaneously meet all of the following criteria:
- It must rank among the top 200 largest public school districts in the United States by enrollment
- It must be the single largest school district within its state (by enrollment)
- It must be located in a state that has at least 9 school districts appearing in the national top 200 ranking
- It must operate more than 190 schools in total

District 2 Requirements:
The second district must simultaneously meet all of the following criteria:
- It must rank among the top 200 largest public school districts in the United States by enrollment
- It must be the second-largest school district within its state (by enrollment)
- The largest school district in the same state must have more than 140,000 students enrolled
- It must have a minority student enrollment of at least 50%

District 3 Requirements:
The third district must simultaneously meet all of the following criteria:
- It must rank among the top 200 largest public school districts in the United States by enrollment
- It must be the third-largest school district within its state (by enrollment)
- It must serve a student population where 100% of students are classified as economically disadvantaged
- It must be located in a state that is NOT part of the southern United States census region (the southern census region includes: Alabama, Arkansas, Delaware, Florida, Georgia, Kentucky, Louisiana, Maryland, Mississippi, North Carolina, Oklahoma, South Carolina, Tennessee, Texas, Virginia, West Virginia, and the District of Columbia)
"""

SOUTHERN_STATES = {
    "Alabama", "Arkansas", "Delaware", "Florida", "Georgia", "Kentucky",
    "Louisiana", "Maryland", "Mississippi", "North Carolina", "Oklahoma",
    "South Carolina", "Tennessee", "Texas", "Virginia", "West Virginia", "District of Columbia"
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DistrictInfo(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    enrollment: Optional[str] = None
    school_count: Optional[str] = None
    minority_enrollment_percent: Optional[str] = None
    economically_disadvantaged_percent: Optional[str] = None
    national_rank: Optional[str] = None

    general_sources: List[str] = Field(default_factory=list)
    enrollment_sources: List[str] = Field(default_factory=list)
    national_ranking_sources: List[str] = Field(default_factory=list)
    state_ranking_sources: List[str] = Field(default_factory=list)
    schools_count_sources: List[str] = Field(default_factory=list)
    largest_state_district_sources: List[str] = Field(default_factory=list)
    demographics_sources: List[str] = Field(default_factory=list)
    economic_sources: List[str] = Field(default_factory=list)
    location_sources: List[str] = Field(default_factory=list)
    state_top200_count_sources: List[str] = Field(default_factory=list)


class DistrictsExtraction(BaseModel):
    district1: Optional[DistrictInfo] = None
    district2: Optional[DistrictInfo] = None
    district3: Optional[DistrictInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_districts() -> str:
    return """
    Extract structured information for exactly three school districts referenced in the answer, aligning the districts to the roles District 1, District 2, and District 3 as described by the task (based on the constraints they appear associated with in the answer).
    For each district, extract:
    - name: Official district name
    - state: The U.S. state where the district is located
    - enrollment: The current student enrollment stated in the answer (string as-written; do not normalize)
    - school_count: The number of schools operated by the district if mentioned
    - minority_enrollment_percent: The minority student enrollment percentage if mentioned
    - economically_disadvantaged_percent: The percentage of students classified as economically disadvantaged if mentioned
    - national_rank: If a national ranking position number is provided (e.g., '#45'), extract it as a string
    - general_sources: All URLs cited that refer to this district generally

    Also extract specialized source URLs if provided (each should be a list of URLs):
    - enrollment_sources: URLs that specifically support enrollment figures for this district
    - national_ranking_sources: URLs confirming appearance in a national "Top 200 largest districts" list
    - state_ranking_sources: URLs showing within-state district enrollment rankings (largest/second-largest/third-largest)
    - schools_count_sources: URLs that support the district's school count
    - largest_state_district_sources: URLs showing the largest district in the same state and its enrollment (used for District 2 threshold requirement)
    - demographics_sources: URLs supporting minority enrollment percentage
    - economic_sources: URLs supporting economically disadvantaged percentage
    - location_sources: URLs confirming the district’s state location
    - state_top200_count_sources: URLs confirming the count of districts in this state appearing in the national top 200 list

    IMPORTANT RULES:
    - Extract only URLs explicitly present in the answer; do not invent URLs.
    - If a specialized category has no URLs, return an empty array for that field.
    - For fields not mentioned, return null (for single values) or empty array (for URL lists).
    - Always return three district objects mapped to district1, district2, district3 in the output.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*lists: List[str]) -> List[str]:
    """Merge and deduplicate URL lists while preserving order."""
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst:
            if not isinstance(url, str):
                continue
            u = url.strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _pick_sources(d: Optional[DistrictInfo], primary: List[str], *fallbacks: List[str]) -> List[str]:
    """Prefer primary sources; fallback to general_sources if no primary provided."""
    if not d:
        return []
    if primary:
        return _merge_sources(primary)
    return _merge_sources(*fallbacks, d.general_sources)


def _safe_name(d: Optional[DistrictInfo]) -> str:
    return (d.name or "").strip()


def _safe_state(d: Optional[DistrictInfo]) -> str:
    return (d.state or "").strip()


def _exists_urls(urls: List[str]) -> bool:
    return bool(urls and len(urls) > 0)


# --------------------------------------------------------------------------- #
# Verification subroutines per district                                       #
# --------------------------------------------------------------------------- #
async def verify_district_1(evaluator: Evaluator, parent_node, d: Optional[DistrictInfo]) -> None:
    district_node = evaluator.add_parallel(
        id="district_1",
        desc="Correctly identify the first district meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # National ranking group
    national_group = evaluator.add_parallel(
        id="district_1_national_ranking",
        desc="District 1 ranks among the top 200 largest U.S. school districts nationally",
        parent=district_node,
        critical=True
    )

    # Enrollment validation
    enroll_group = evaluator.add_parallel(
        id="district_1_enrollment_validation",
        desc="Confirm District 1's enrollment number",
        parent=national_group,
        critical=True
    )
    # Sources existence
    enroll_sources = _pick_sources(d, d.enrollment_sources if d else [], d.general_sources if d else [])
    evaluator.add_custom_node(
        result=_exists_urls(enroll_sources),
        id="district_1_enrollment_reference",
        desc="Provide reference URL for District 1's enrollment",
        parent=enroll_group,
        critical=True
    )
    # Enrollment claim
    enroll_leaf = evaluator.add_leaf(
        id="district_1_enrollment_validation_check",
        desc="Enrollment value matches referenced sources",
        parent=enroll_group,
        critical=True
    )
    name1, state1, enrollment1 = _safe_name(d), _safe_state(d), (d.enrollment or "").strip() if d else ("")
    claim_enroll = f"The current student enrollment of {name1} in {state1} is {enrollment1}."
    await evaluator.verify(
        claim=claim_enroll,
        node=enroll_leaf,
        sources=enroll_sources,
        additional_instruction="Use the provided source(s) to verify the stated enrollment. Minor rounding differences or very recent updates are acceptable if clearly consistent."
    )

    # National list verification
    nat_list_group = evaluator.add_parallel(
        id="district_1_national_list_verification",
        desc="Verify District 1 appears in the national top 200 list",
        parent=national_group,
        critical=True
    )
    nat_sources = _pick_sources(d, d.national_ranking_sources if d else [], d.general_sources if d else [])
    evaluator.add_custom_node(
        result=_exists_urls(nat_sources),
        id="district_1_national_reference",
        desc="Provide reference URL confirming national ranking",
        parent=nat_list_group,
        critical=True
    )
    nat_list_leaf = evaluator.add_leaf(
        id="district_1_national_list_verification_check",
        desc="District appears in national Top 200 list",
        parent=nat_list_group,
        critical=True
    )
    claim_nat = f"{name1} appears in a national 'Top 200 largest U.S. public school districts by enrollment' list."
    await evaluator.verify(
        claim=claim_nat,
        node=nat_list_leaf,
        sources=nat_sources,
        additional_instruction="Confirm that the district is included in a national list of the top 200 largest districts by enrollment."
    )

    # State largest
    state_largest = evaluator.add_parallel(
        id="district_1_state_largest",
        desc="District 1 is the largest school district in its state",
        parent=district_node,
        critical=True
    )
    state_comp = evaluator.add_parallel(
        id="district_1_state_comparison",
        desc="Confirm District 1 has higher enrollment than all other districts in its state",
        parent=state_largest,
        critical=True
    )
    state_rank_sources = _pick_sources(d, d.state_ranking_sources if d else [], d.general_sources if d else [])
    evaluator.add_custom_node(
        result=_exists_urls(state_rank_sources),
        id="district_1_state_ranking_reference",
        desc="Provide reference URL for state district rankings",
        parent=state_comp,
        critical=True
    )
    largest_leaf = evaluator.add_leaf(
        id="district_1_state_largest_check",
        desc="Largest-by-enrollment in its state",
        parent=state_comp,
        critical=True
    )
    claim_largest = f"Within the state of {state1}, {name1} is the largest public school district by enrollment."
    await evaluator.verify(
        claim=claim_largest,
        node=largest_leaf,
        sources=state_rank_sources,
        additional_instruction="Use state-level district ranking or enrollment comparison sources to confirm this district is the largest by enrollment."
    )

    # State has at least 9 districts in top 200
    state_count_group = evaluator.add_parallel(
        id="district_1_state_count",
        desc="District 1's state has at least 9 districts in the national top 200",
        parent=district_node,
        critical=True
    )
    state_count_sources = _pick_sources(d, d.state_top200_count_sources if d else [], d.general_sources if d else [])
    evaluator.add_custom_node(
        result=_exists_urls(state_count_sources),
        id="district_1_state_count_reference",
        desc="Provide reference URL listing districts by state in top 200",
        parent=state_count_group,
        critical=True
    )
    state_count_leaf = evaluator.add_leaf(
        id="district_1_state_count_check",
        desc="State has ≥ 9 districts in Top 200",
        parent=state_count_group,
        critical=True
    )
    claim_state_count = f"The state of {state1} has at least 9 school districts appearing in the national top 200 ranking by enrollment."
    await evaluator.verify(
        claim=claim_state_count,
        node=state_count_leaf,
        sources=state_count_sources,
        additional_instruction="Verify the count of the state's districts appearing in the national Top 200 list; aggregated reports or filtered lists by state are acceptable."
    )

    # School count > 190
    school_count_group = evaluator.add_parallel(
        id="district_1_school_count",
        desc="District 1 operates more than 190 schools",
        parent=district_node,
        critical=True
    )
    schools_sources = _pick_sources(d, d.schools_count_sources if d else [], d.general_sources if d else [])
    evaluator.add_custom_node(
        result=_exists_urls(schools_sources),
        id="district_1_schools_reference",
        desc="Provide reference URL for District 1's school count",
        parent=school_count_group,
        critical=True
    )
    schools_leaf = evaluator.add_leaf(
        id="district_1_school_count_check",
        desc="Operates more than 190 schools",
        parent=school_count_group,
        critical=True
    )
    if d and d.school_count:
        claim_schools = f"{name1} operates {d.school_count} schools, which is more than 190."
    else:
        claim_schools = f"{name1} operates more than 190 schools in total."
    await evaluator.verify(
        claim=claim_schools,
        node=schools_leaf,
        sources=schools_sources,
        additional_instruction="Confirm the district's total number of schools exceeds 190. Accept clearly supported counts (e.g., official fact pages)."
    )


async def verify_district_2(evaluator: Evaluator, parent_node, d: Optional[DistrictInfo]) -> None:
    district_node = evaluator.add_parallel(
        id="district_2",
        desc="Correctly identify the second district meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # National ranking group
    national_group = evaluator.add_parallel(
        id="district_2_national_ranking",
        desc="District 2 ranks among the top 200 largest U.S. school districts nationally",
        parent=district_node,
        critical=True
    )

    # Enrollment validation
    enroll_group = evaluator.add_parallel(
        id="district_2_enrollment_validation",
        desc="Confirm District 2's enrollment number",
        parent=national_group,
        critical=True
    )
    enroll_sources = _pick_sources(d, d.enrollment_sources if d else [], d.general_sources if d else [])
    evaluator.add_custom_node(
        result=_exists_urls(enroll_sources),
        id="district_2_enrollment_reference",
        desc="Provide reference URL for District 2's enrollment",
        parent=enroll_group,
        critical=True
    )
    enroll_leaf = evaluator.add_leaf(
        id="district_2_enrollment_validation_check",
        desc="Enrollment value matches referenced sources",
        parent=enroll_group,
        critical=True
    )
    name2, state2, enrollment2 = _safe_name(d), _safe_state(d), (d.enrollment or "").strip() if d else ("")
    claim_enroll = f"The current student enrollment of {name2} in {state2} is {enrollment2}."
    await evaluator.verify(
        claim=claim_enroll,
        node=enroll_leaf,
        sources=enroll_sources,
        additional_instruction="Use the provided source(s) to verify the stated enrollment. Minor rounding differences or very recent updates are acceptable if clearly consistent."
    )

    # National list verification
    nat_list_group = evaluator.add_parallel(
        id="district_2_national_list_verification",
        desc="Verify District 2 appears in the national top 200 list",
        parent=national_group,
        critical=True
    )
    nat_sources = _pick_sources(d, d.national_ranking_sources if d else [], d.general_sources if d else [])
    evaluator.add_custom_node(
        result=_exists_urls(nat_sources),
        id="district_2_national_reference",
        desc="Provide reference URL confirming national ranking",
        parent=nat_list_group,
        critical=True
    )
    nat_list_leaf = evaluator.add_leaf(
        id="district_2_national_list_verification_check",
        desc="District appears in national Top 200 list",
        parent=nat_list_group,
        critical=True
    )
    claim_nat = f"{name2} appears in a national 'Top 200 largest U.S. public school districts by enrollment' list."
    await evaluator.verify(
        claim=claim_nat,
        node=nat_list_leaf,
        sources=nat_sources,
        additional_instruction="Confirm that the district is included in a national list of the top 200 largest districts by enrollment."
    )

    # State second-largest
    state_second = evaluator.add_parallel(
        id="district_2_state_second",
        desc="District 2 is the second-largest school district in its state",
        parent=district_node,
        critical=True
    )
    state_comp = evaluator.add_parallel(
        id="district_2_state_comparison",
        desc="Confirm District 2 ranks second in enrollment within its state",
        parent=state_second,
        critical=True
    )
    state_rank_sources = _pick_sources(d, d.state_ranking_sources if d else [], d.general_sources if d else [])
    evaluator.add_custom_node(
        result=_exists_urls(state_rank_sources),
        id="district_2_state_ranking_reference",
        desc="Provide reference URL for state district rankings",
        parent=state_comp,
        critical=True
    )
    second_leaf = evaluator.add_leaf(
        id="district_2_state_second_check",
        desc="Second-by-enrollment in its state",
        parent=state_comp,
        critical=True
    )
    claim_second = f"Within the state of {state2}, {name2} is the second-largest public school district by enrollment."
    await evaluator.verify(
        claim=claim_second,
        node=second_leaf,
        sources=state_rank_sources,
        additional_instruction="Use state-level district ranking or enrollment comparison sources to confirm this district is second by enrollment."
    )

    # Largest threshold in same state > 140,000
    largest_thresh = evaluator.add_parallel(
        id="district_2_largest_threshold",
        desc="The largest district in District 2's state has more than 140,000 students",
        parent=district_node,
        critical=True
    )
    largest_sources = _pick_sources(d, d.largest_state_district_sources if d else [], d.general_sources if d else [])
    evaluator.add_custom_node(
        result=_exists_urls(largest_sources),
        id="district_2_largest_reference",
        desc="Provide reference URL for the largest district's enrollment in District 2's state",
        parent=largest_thresh,
        critical=True
    )
    largest_leaf = evaluator.add_leaf(
        id="district_2_largest_threshold_check",
        desc="Largest district in state has > 140,000 enrollment",
        parent=largest_thresh,
        critical=True
    )
    claim_largest = f"The largest public school district in {state2} has more than 140,000 students enrolled."
    await evaluator.verify(
        claim=claim_largest,
        node=largest_leaf,
        sources=largest_sources,
        additional_instruction="Verify that the largest district in the same state exceeds 140,000 enrollment. The source should clearly state the enrollment for the top district."
    )

    # Minority enrollment >= 50%
    minority_group = evaluator.add_parallel(
        id="district_2_minority_enrollment",
        desc="District 2 has minority enrollment of at least 50%",
        parent=district_node,
        critical=True
    )
    demo_sources = _pick_sources(d, d.demographics_sources if d else [], d.general_sources if d else [])
    evaluator.add_custom_node(
        result=_exists_urls(demo_sources),
        id="district_2_demographics_reference",
        desc="Provide reference URL for District 2's demographic data",
        parent=minority_group,
        critical=True
    )
    minority_leaf = evaluator.add_leaf(
        id="district_2_minority_enrollment_check",
        desc="Minority enrollment ≥ 50%",
        parent=minority_group,
        critical=True
    )
    if d and d.minority_enrollment_percent:
        claim_minority = f"The minority student enrollment of {name2} is {d.minority_enrollment_percent}, which is at least 50%."
    else:
        claim_minority = f"The minority student enrollment of {name2} is at least 50%."
    await evaluator.verify(
        claim=claim_minority,
        node=minority_leaf,
        sources=demo_sources,
        additional_instruction="Confirm the minority student enrollment is at least 50%. Accept official report PDFs, dashboards, or district/state data pages."
    )


async def verify_district_3(evaluator: Evaluator, parent_node, d: Optional[DistrictInfo]) -> None:
    district_node = evaluator.add_parallel(
        id="district_3",
        desc="Correctly identify the third district meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # National ranking group
    national_group = evaluator.add_parallel(
        id="district_3_national_ranking",
        desc="District 3 ranks among the top 200 largest U.S. school districts nationally",
        parent=district_node,
        critical=True
    )

    # Enrollment validation
    enroll_group = evaluator.add_parallel(
        id="district_3_enrollment_validation",
        desc="Confirm District 3's enrollment number",
        parent=national_group,
        critical=True
    )
    enroll_sources = _pick_sources(d, d.enrollment_sources if d else [], d.general_sources if d else [])
    evaluator.add_custom_node(
        result=_exists_urls(enroll_sources),
        id="district_3_enrollment_reference",
        desc="Provide reference URL for District 3's enrollment",
        parent=enroll_group,
        critical=True
    )
    enroll_leaf = evaluator.add_leaf(
        id="district_3_enrollment_validation_check",
        desc="Enrollment value matches referenced sources",
        parent=enroll_group,
        critical=True
    )
    name3, state3, enrollment3 = _safe_name(d), _safe_state(d), (d.enrollment or "").strip() if d else ("")
    claim_enroll = f"The current student enrollment of {name3} in {state3} is {enrollment3}."
    await evaluator.verify(
        claim=claim_enroll,
        node=enroll_leaf,
        sources=enroll_sources,
        additional_instruction="Use the provided source(s) to verify the stated enrollment. Minor rounding differences or very recent updates are acceptable if clearly consistent."
    )

    # National list verification
    nat_list_group = evaluator.add_parallel(
        id="district_3_national_list_verification",
        desc="Verify District 3 appears in the national top 200 list",
        parent=national_group,
        critical=True
    )
    nat_sources = _pick_sources(d, d.national_ranking_sources if d else [], d.general_sources if d else [])
    evaluator.add_custom_node(
        result=_exists_urls(nat_sources),
        id="district_3_national_reference",
        desc="Provide reference URL confirming national ranking",
        parent=nat_list_group,
        critical=True
    )
    nat_list_leaf = evaluator.add_leaf(
        id="district_3_national_list_verification_check",
        desc="District appears in national Top 200 list",
        parent=nat_list_group,
        critical=True
    )
    claim_nat = f"{name3} appears in a national 'Top 200 largest U.S. public school districts by enrollment' list."
    await evaluator.verify(
        claim=claim_nat,
        node=nat_list_leaf,
        sources=nat_sources,
        additional_instruction="Confirm that the district is included in a national list of the top 200 largest districts by enrollment."
    )

    # State third-largest
    state_third = evaluator.add_parallel(
        id="district_3_state_third",
        desc="District 3 is the third-largest school district in its state",
        parent=district_node,
        critical=True
    )
    state_comp = evaluator.add_parallel(
        id="district_3_state_comparison",
        desc="Confirm District 3 ranks third in enrollment within its state",
        parent=state_third,
        critical=True
    )
    state_rank_sources = _pick_sources(d, d.state_ranking_sources if d else [], d.general_sources if d else [])
    evaluator.add_custom_node(
        result=_exists_urls(state_rank_sources),
        id="district_3_state_ranking_reference",
        desc="Provide reference URL for state district rankings",
        parent=state_comp,
        critical=True
    )
    third_leaf = evaluator.add_leaf(
        id="district_3_state_third_check",
        desc="Third-by-enrollment in its state",
        parent=state_comp,
        critical=True
    )
    claim_third = f"Within the state of {state3}, {name3} is the third-largest public school district by enrollment."
    await evaluator.verify(
        claim=claim_third,
        node=third_leaf,
        sources=state_rank_sources,
        additional_instruction="Use state-level district ranking or enrollment comparison sources to confirm this district is third by enrollment."
    )

    # 100% economically disadvantaged
    econ_group = evaluator.add_parallel(
        id="district_3_economically_disadvantaged",
        desc="District 3 serves a student population where 100% are economically disadvantaged",
        parent=district_node,
        critical=True
    )
    econ_sources = _pick_sources(d, d.economic_sources if d else [], d.general_sources if d else [])
    evaluator.add_custom_node(
        result=_exists_urls(econ_sources),
        id="district_3_economic_reference",
        desc="Provide reference URL for District 3's economically disadvantaged percentage",
        parent=econ_group,
        critical=True
    )
    econ_leaf = evaluator.add_leaf(
        id="district_3_economically_disadvantaged_check",
        desc="Economically disadvantaged = 100%",
        parent=econ_group,
        critical=True
    )
    if d and d.economically_disadvantaged_percent:
        claim_econ = f"In {name3}, {d.economically_disadvantaged_percent} of students are economically disadvantaged, which equals 100%."
    else:
        claim_econ = f"In {name3}, 100% of students are economically disadvantaged."
    await evaluator.verify(
        claim=claim_econ,
        node=econ_leaf,
        sources=econ_sources,
        additional_instruction="Verify the economically disadvantaged percentage equals 100%. Accept official reports or data dashboards that explicitly state this value."
    )

    # Non-southern census region
    non_south_group = evaluator.add_parallel(
        id="district_3_non_southern",
        desc="District 3 is located in a state not in the southern U.S. census region",
        parent=district_node,
        critical=True
    )
    loc_sources = _pick_sources(d, d.location_sources if d else [], d.general_sources if d else [])
    evaluator.add_custom_node(
        result=_exists_urls(loc_sources),
        id="district_3_location_reference",
        desc="Provide reference URL confirming District 3's state",
        parent=non_south_group,
        critical=True
    )
    non_south_leaf = evaluator.add_leaf(
        id="district_3_non_southern_check",
        desc="State not in Southern Census Region",
        parent=non_south_group,
        critical=True
    )
    state3_val = state3
    claim_non_south = f"The state '{state3_val}' is not part of the southern United States census region."
    await evaluator.verify(
        claim=claim_non_south,
        node=non_south_leaf,
        sources=None,  # Pure logical check; evidence for state is provided by location_reference
        additional_instruction=(
            "Use the provided list of southern census region states to check membership. "
            "Southern states include: Alabama, Arkansas, Delaware, Florida, Georgia, Kentucky, Louisiana, Maryland, "
            "Mississippi, North Carolina, Oklahoma, South Carolina, Tennessee, Texas, Virginia, West Virginia, and the District of Columbia."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the U.S. school districts ranking/demographics task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel aggregation at top-level
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

    # Extract district information
    extracted = await evaluator.extract(
        prompt=prompt_extract_districts(),
        template_class=DistrictsExtraction,
        extraction_name="districts_extraction"
    )

    # Create a task completion node (non-critical to allow partial credit)
    task_node = evaluator.add_parallel(
        id="task_completion",
        desc="Successfully identify all three school districts meeting their respective combined requirements",
        parent=root,
        critical=False
    )

    # Add custom info for southern states reference
    evaluator.add_custom_info(
        info={"southern_census_states": sorted(list(SOUTHERN_STATES))},
        info_type="reference",
        info_name="southern_states_list"
    )

    # Verify each district according to its specific constraints
    await verify_district_1(evaluator, task_node, extracted.district1)
    await verify_district_2(evaluator, task_node, extracted.district2)
    await verify_district_3(evaluator, task_node, extracted.district3)

    # Return structured evaluation summary
    return evaluator.get_summary()