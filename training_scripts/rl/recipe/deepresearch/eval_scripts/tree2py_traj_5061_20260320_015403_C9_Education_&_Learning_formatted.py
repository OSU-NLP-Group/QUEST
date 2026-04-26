import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "largest_us_school_districts_2023_2024_constraints"
TASK_DESCRIPTION = """
Among the top 20 largest public school districts in the United States for the 2023-2024 school year, identify the four school districts that meet the following specific criteria:

1. First District: This district must be located in a state that belongs to the South Atlantic Census Division, specifically in Florida. It must be ranked between 6th and 10th nationally among the largest school districts, and have an enrollment between 240,000 and 260,000 students (inclusive).

2. Second District: This district must be located in a state that belongs to the West South Central Census Division, specifically in Texas. It must be ranked between 11th and 20th nationally among the largest school districts, and have an enrollment between 135,000 and 145,000 students (inclusive).

3. Third District: This district must be located in a state that belongs to the Mountain Census Division (which includes Arizona, Colorado, Idaho, Montana, Nevada, New Mexico, Utah, and Wyoming). It must be ranked between 1st and 10th nationally among the largest school districts, and have an enrollment exceeding 300,000 students.

4. Fourth District: This district must be located in the state of Maryland, which belongs to the South Atlantic Census Division. It must be ranked between 11th and 20th nationally among the largest school districts, and have an enrollment between 159,000 and 162,000 students (inclusive).

For each district, provide:
- The full official name of the school district
- The exact enrollment number for 2023-2024
- The national ranking position
- The state and county/city location
- The U.S. Census Division
- Reference URLs confirming all key information

Note: Use the official U.S. Census Bureau classification of states into Census Regions and Divisions when determining Census Division membership.
"""

MOUNTAIN_STATES = {"Arizona", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico", "Utah", "Wyoming"}
DIVISION_BY_STATE = {
    # South Atlantic
    "Delaware": "South Atlantic", "Florida": "South Atlantic", "Georgia": "South Atlantic",
    "Maryland": "South Atlantic", "North Carolina": "South Atlantic", "South Carolina": "South Atlantic",
    "Virginia": "South Atlantic", "West Virginia": "South Atlantic", "District of Columbia": "South Atlantic",
    # West South Central
    "Arkansas": "West South Central", "Louisiana": "West South Central",
    "Oklahoma": "West South Central", "Texas": "West South Central",
    # Mountain
    "Arizona": "Mountain", "Colorado": "Mountain", "Idaho": "Mountain", "Montana": "Mountain",
    "Nevada": "Mountain", "New Mexico": "Mountain", "Utah": "Mountain", "Wyoming": "Mountain",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DistrictEntry(BaseModel):
    name: Optional[str] = None
    name_sources: List[str] = Field(default_factory=list)

    state: Optional[str] = None
    state_sources: List[str] = Field(default_factory=list)

    county_city: Optional[str] = None
    county_city_sources: List[str] = Field(default_factory=list)

    census_division: Optional[str] = None
    census_sources: List[str] = Field(default_factory=list)

    ranking: Optional[str] = None
    ranking_sources: List[str] = Field(default_factory=list)

    enrollment: Optional[str] = None
    enrollment_sources: List[str] = Field(default_factory=list)


class DistrictsExtraction(BaseModel):
    district1: Optional[DistrictEntry] = None
    district2: Optional[DistrictEntry] = None
    district3: Optional[DistrictEntry] = None
    district4: Optional[DistrictEntry] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_districts() -> str:
    return """
    Extract structured information for exactly four districts as presented in the answer, corresponding to:
    - district1: the FIRST required district (Florida, rank 6–10, enrollment 240,000–260,000 inclusive)
    - district2: the SECOND required district (Texas, rank 11–20, enrollment 135,000–145,000 inclusive)
    - district3: the THIRD required district (state in Mountain Division, rank 1–10, enrollment >300,000)
    - district4: the FOURTH required district (Maryland, rank 11–20, enrollment 159,000–162,000 inclusive)

    For each districtN, extract:
    - name: full official district name as written in the answer
    - name_sources: list of all URLs in the answer that support/confirm the official name
    - state: state as written (e.g., Florida)
    - state_sources: list of URLs supporting the stated state location of the district
    - county_city: county or city location as written (e.g., Broward County or Clark County / City of X)
    - county_city_sources: list of URLs supporting the county/city location
    - census_division: U.S. Census Division name as written (e.g., South Atlantic, Mountain, West South Central)
    - census_sources: list of URLs supporting the Census Division classification for the state
    - ranking: the national ranking position among the largest U.S. public school districts for the 2023–2024 school year, as written
    - ranking_sources: list of URLs supporting the ranking (should be for the 2023–2024 school year)
    - enrollment: the exact enrollment number for 2023–2024 as written (keep any commas)
    - enrollment_sources: list of URLs supporting the 2023–2024 enrollment

    Rules:
    - Only extract values explicitly present in the answer. Do not infer or invent anything.
    - For each field with URLs, include all URLs that the answer explicitly cites for that field; if none are cited, return an empty list.
    - Keep numbers as strings exactly as written (e.g., "160,123").
    - If any field is missing in the answer, set that field to null and the corresponding sources array to an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(urls: Optional[List[str]]) -> List[str]:
    return [u.strip() for u in (urls or []) if isinstance(u, str) and u.strip()]


def _exists(value: Optional[str]) -> bool:
    return value is not None and str(value).strip() != ""


def _name_or_placeholder(entry: Optional[DistrictEntry]) -> str:
    return (entry.name or "the school district").strip() if entry else "the school district"


# --------------------------------------------------------------------------- #
# Verification for a single district                                          #
# --------------------------------------------------------------------------- #
async def verify_single_district(
    evaluator: Evaluator,
    parent_node,
    entry: Optional[DistrictEntry],
    district_index: int,
    *,
    block_desc: str,
    expected_state: Optional[str] = None,
    expected_division: Optional[str] = None,
    require_state_in_mountain: bool = False,
    ranking_range: Optional[Tuple[int, int]] = None,  # inclusive bounds
    enrollment_range: Optional[Tuple[int, int]] = None,  # inclusive bounds
    enrollment_exceeds: Optional[int] = None,
) -> None:
    """
    Build verification subtree for one district based on constraints.
    """
    # District-level parallel container (non-critical, allows partial credit across districts)
    block = evaluator.add_parallel(
        id=f"district_{district_index}",
        desc=block_desc,
        parent=parent_node,
        critical=False,
    )

    # Defensive defaults
    name = entry.name.strip() if entry and entry.name else ""
    state_value = entry.state.strip() if entry and entry.state else ""
    county_city_value = entry.county_city.strip() if entry and entry.county_city else ""
    census_value = entry.census_division.strip() if entry and entry.census_division else ""
    ranking_value = entry.ranking.strip() if entry and entry.ranking else ""
    enrollment_value = entry.enrollment.strip() if entry and entry.enrollment else ""

    name_sources = _safe_list(entry.name_sources if entry else [])
    state_sources = _safe_list(entry.state_sources if entry else [])
    county_city_sources = _safe_list(entry.county_city_sources if entry else [])
    census_sources = _safe_list(entry.census_sources if entry else [])
    ranking_sources = _safe_list(entry.ranking_sources if entry else [])
    enrollment_sources = _safe_list(entry.enrollment_sources if entry else [])

    # 1) Name verification bundle (critical)
    name_node = evaluator.add_parallel(
        id=f"district_{district_index}_name",
        desc="Provide the full official name of the school district",
        parent=block,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_exists(name),
        id=f"district_{district_index}_name_exists",
        desc="District official name is provided in the answer",
        parent=name_node,
        critical=True,
    )
    # Require at least one source before attempting verification
    name_src_exist = evaluator.add_custom_node(
        result=len(name_sources) > 0,
        id=f"district_{district_index}_name_sources_present",
        desc="At least one URL is cited to support the official name",
        parent=name_node,
        critical=True,
    )
    # Actual verification (blocked if no sources)
    name_url_leaf = evaluator.add_leaf(
        id=f"district_{district_index}_name_url",
        desc="URL reference confirming the official district name",
        parent=name_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The official name of the school district is '{name}'.",
        node=name_url_leaf,
        sources=name_sources,
        additional_instruction="Verify the exact or commonly accepted official name on the cited page(s). Minor variants or acronyms are acceptable if the page clearly indicates the official name.",
    )

    # 2) State verification bundle (critical)
    state_node = evaluator.add_parallel(
        id=f"district_{district_index}_state",
        desc="Verify the state location of the district",
        parent=block,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_exists(state_value) or (expected_state is not None),
        id=f"district_{district_index}_state_value_present",
        desc="A target state is known (either extracted from the answer or specified by the task constraints)",
        parent=state_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(state_sources) > 0,
        id=f"district_{district_index}_state_sources_present",
        desc="At least one URL is cited to support the district's state location",
        parent=state_node,
        critical=True,
    )
    # Build the state claim anchored on district name to avoid irrelevant matches
    target_state = expected_state or state_value
    state_url_leaf = evaluator.add_leaf(
        id=f"district_{district_index}_state_url",
        desc=("URL reference confirming "
              f"{(expected_state or state_value) or 'the'} location"),
        parent=state_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The school district '{name}' is located in the state of {target_state}.",
        node=state_url_leaf,
        sources=state_sources,
        additional_instruction="Confirm that the cited page clearly places this district in the specified U.S. state.",
    )

    # 3) County/City verification bundle (critical)
    county_node = evaluator.add_parallel(
        id=f"district_{district_index}_county_city",
        desc="Provide the county or city location of the district",
        parent=block,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_exists(county_city_value),
        id=f"district_{district_index}_county_city_value_present",
        desc="County/City value is provided in the answer",
        parent=county_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(county_city_sources) > 0,
        id=f"district_{district_index}_county_city_sources_present",
        desc="At least one URL is cited to support the county/city location",
        parent=county_node,
        critical=True,
    )
    county_city_leaf = evaluator.add_leaf(
        id=f"district_{district_index}_county_city_url",
        desc="URL reference confirming county/city location",
        parent=county_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The school district '{name}' is associated with or operates in {county_city_value}.",
        node=county_city_leaf,
        sources=county_city_sources,
        additional_instruction="Accept pages that clearly indicate the district serves, is headquartered in, or corresponds to the stated county or city.",
    )

    # 4) Census Division verification bundle (critical)
    census_node = evaluator.add_parallel(
        id=f"district_{district_index}_census_division",
        desc="Verify the state's Census Division classification",
        parent=block,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(census_sources) > 0,
        id=f"district_{district_index}_census_sources_present",
        desc="At least one URL is cited to support Census Division classification",
        parent=census_node,
        critical=True,
    )

    # Determine census division claim depending on district spec
    if require_state_in_mountain:
        # Use extracted state to assert Mountain membership
        census_leaf = evaluator.add_leaf(
            id=f"district_{district_index}_census_url",
            desc="URL reference confirming Census Division classification (Mountain)",
            parent=census_node,
            critical=True,
        )
        await evaluator.verify(
            claim=f"The state of {state_value} belongs to the Mountain Census Division (AZ, CO, ID, MT, NV, NM, UT, WY).",
            node=census_leaf,
            sources=census_sources,
            additional_instruction="Rely on official U.S. Census Bureau materials or reputable summaries showing that this state is in the Mountain Division.",
        )
    else:
        # Use expected_division (e.g., South Atlantic, West South Central)
        division = expected_division or census_value
        census_leaf = evaluator.add_leaf(
            id=f"district_{district_index}_census_url",
            desc="URL reference confirming Census Division classification",
            parent=census_node,
            critical=True,
        )
        await evaluator.verify(
            claim=f"The state of {target_state} belongs to the {division} Census Division.",
            node=census_leaf,
            sources=census_sources,
            additional_instruction="Use the official U.S. Census Bureau classification of regions and divisions to verify this.",
        )

    # 5) Ranking verification bundle (critical)
    ranking_node = evaluator.add_parallel(
        id=f"district_{district_index}_ranking",
        desc="Verify the district's national ranking among largest U.S. public school districts (2023–2024)",
        parent=block,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(ranking_sources) > 0,
        id=f"district_{district_index}_ranking_sources_present",
        desc="At least one URL is cited to support the 2023–2024 national ranking",
        parent=ranking_node,
        critical=True,
    )
    ranking_leaf = evaluator.add_leaf(
        id=f"district_{district_index}_ranking_url",
        desc="URL reference confirming district ranking",
        parent=ranking_node,
        critical=True,
    )
    if ranking_range:
        lo, hi = ranking_range
        ranking_claim = (
            f"For the 2023–2024 school year, the school district '{name}' is ranked between {lo} and {hi} "
            f"(inclusive) among the largest U.S. public school districts by enrollment."
        )
    else:
        # Fallback generic statement if no explicit range provided (should not happen for this task)
        ranking_claim = (
            f"For the 2023–2024 school year, the school district '{name}' has the stated national ranking among the largest U.S. public school districts."
        )
    await evaluator.verify(
        claim=ranking_claim,
        node=ranking_leaf,
        sources=ranking_sources,
        additional_instruction="The cited page should clearly show the national rank (or a list where the district appears with a rank) for the 2023–2024 school year.",
    )

    # 6) Enrollment verification bundle (critical)
    enrollment_node = evaluator.add_parallel(
        id=f"district_{district_index}_enrollment",
        desc="Verify the district's 2023–2024 enrollment and whether it meets the required constraint",
        parent=block,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(enrollment_sources) > 0,
        id=f"district_{district_index}_enrollment_sources_present",
        desc="At least one URL is cited to support the 2023–2024 enrollment",
        parent=enrollment_node,
        critical=True,
    )
    enrollment_leaf = evaluator.add_leaf(
        id=f"district_{district_index}_enrollment_url",
        desc="URL reference confirming enrollment number",
        parent=enrollment_node,
        critical=True,
    )
    if enrollment_range:
        lo, hi = enrollment_range
        enrollment_claim = (
            f"For the 2023–2024 school year, the school district '{name}' has an enrollment between {lo:,} and {hi:,} students (inclusive)."
        )
    elif enrollment_exceeds is not None:
        enrollment_claim = (
            f"For the 2023–2024 school year, the school district '{name}' has an enrollment exceeding {enrollment_exceeds:,} students."
        )
    else:
        enrollment_claim = (
            f"For the 2023–2024 school year, the school district '{name}' has the stated enrollment."
        )
    await evaluator.verify(
        claim=enrollment_claim,
        node=enrollment_leaf,
        sources=enrollment_sources,
        additional_instruction="Confirm that the cited page lists a 2023–2024 enrollment consistent with the claim. Reasonable rounding or formatting differences (commas) are acceptable.",
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the largest U.S. school districts (2023–2024) constrained-identification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent district blocks
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
    # IMPORTANT: Root kept non-critical to allow partial credit across districts
    # even though the original rubric marks root as critical.

    # Extract structured district data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_districts(),
        template_class=DistrictsExtraction,
        extraction_name="districts_extraction",
    )

    # Record helpful ground-truth info for reference (not used for gating)
    evaluator.add_ground_truth(
        {
            "census_division_by_state_snapshot": DIVISION_BY_STATE,
            "mountain_states": sorted(list(MOUNTAIN_STATES)),
            "evaluation_year": "2023–2024",
            "notes": "Verification relies on URLs cited by the answer; the LLM judge cross-checks claims against those URLs."
        },
        gt_type="reference_context",
    )

    # Build each of the four district verification subtrees
    # District 1: Florida, South Atlantic; rank 6–10; enrollment 240k–260k
    await verify_single_district(
        evaluator,
        root,
        getattr(extracted, "district1"),
        district_index=1,
        block_desc="First required district: Located in South Atlantic Division, Florida, ranked 6–10, enrollment 240,000–260,000",
        expected_state="Florida",
        expected_division="South Atlantic",
        require_state_in_mountain=False,
        ranking_range=(6, 10),
        enrollment_range=(240_000, 260_000),
        enrollment_exceeds=None,
    )

    # District 2: Texas, West South Central; rank 11–20; enrollment 135k–145k
    await verify_single_district(
        evaluator,
        root,
        getattr(extracted, "district2"),
        district_index=2,
        block_desc="Second required district: Located in West South Central Division, Texas, ranked 11–20, enrollment 135,000–145,000",
        expected_state="Texas",
        expected_division="West South Central",
        require_state_in_mountain=False,
        ranking_range=(11, 20),
        enrollment_range=(135_000, 145_000),
        enrollment_exceeds=None,
    )

    # District 3: Mountain Division state; rank 1–10; enrollment >300k
    await verify_single_district(
        evaluator,
        root,
        getattr(extracted, "district3"),
        district_index=3,
        block_desc="Third required district: Located in Mountain Census Division, ranked 1–10, enrollment above 300,000",
        expected_state=None,
        expected_division=None,
        require_state_in_mountain=True,
        ranking_range=(1, 10),
        enrollment_range=None,
        enrollment_exceeds=300_000,
    )

    # District 4: Maryland (South Atlantic); rank 11–20; enrollment 159k–162k
    await verify_single_district(
        evaluator,
        root,
        getattr(extracted, "district4"),
        district_index=4,
        block_desc="Fourth required district: Located in Maryland (South Atlantic Division), ranked 11–20, enrollment 159,000–162,000",
        expected_state="Maryland",
        expected_division="South Atlantic",
        require_state_in_mountain=False,
        ranking_range=(11, 20),
        enrollment_range=(159_000, 162_000),
        enrollment_exceeds=None,
    )

    # Return structured evaluation summary
    return evaluator.get_summary()