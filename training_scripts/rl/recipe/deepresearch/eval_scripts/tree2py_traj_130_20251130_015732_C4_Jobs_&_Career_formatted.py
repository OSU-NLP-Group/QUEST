import asyncio
import logging
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "big_ten_pa_nj_ad"
TASK_DESCRIPTION = (
    "Identifies Big Ten Conference schools with main campuses in Pennsylvania or New Jersey for the 2024-2025 season "
    "and provides required information about each school's location and athletic director"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PennsylvaniaSchoolInfo(BaseModel):
    school_name: Optional[str] = None
    campus_location: Optional[str] = None
    athletic_director_name: Optional[str] = None
    athletic_director_start_year: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class NewJerseySchoolInfo(BaseModel):
    school_name: Optional[str] = None
    campus_location: Optional[str] = None
    athletic_director_status: Optional[str] = None  # Expect "permanent" or "interim"
    interim_athletic_director_name: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class PA_NJ_Extraction(BaseModel):
    pennsylvania: Optional[PennsylvaniaSchoolInfo] = None
    new_jersey: Optional[NewJerseySchoolInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pa_nj() -> str:
    return """
    Extract structured information about Big Ten Conference schools with main campuses in Pennsylvania and New Jersey for the 2024-2025 season from the provided answer.

    Return a JSON object with two top-level keys: "pennsylvania" and "new_jersey".
    For "pennsylvania", extract:
      - school_name: The Big Ten school whose main campus is in Pennsylvania.
      - campus_location: The specific city or cities where this Pennsylvania school's main campus is located (return exactly as stated in the answer, e.g., "University Park" or "State College, PA").
      - athletic_director_name: The full name of this Pennsylvania school's current athletic director.
      - athletic_director_start_year: The year this person began serving as athletic director at this school (year only; if a month/day is given, extract just the year as a string).
      - source_urls: All URLs cited in the answer that support any of the above Pennsylvania-related claims. Extract actual URLs only, including those embedded in markdown links.

    For "new_jersey", extract:
      - school_name: The Big Ten school whose main campus is in New Jersey.
      - campus_location: The specific city or cities where this New Jersey school's main campus is located (return exactly as stated in the answer).
      - athletic_director_status: Either "permanent" or "interim". Normalize to lowercase. If unclear or not provided, return null.
      - interim_athletic_director_name: If the status is interim, provide the full name of the interim athletic director; otherwise return null.
      - source_urls: All URLs cited in the answer that support any of the above New Jersey-related claims. Extract actual URLs only, including those embedded in markdown links.

    General rules:
    - Do not invent information. If a field is not present in the answer, return null for the field.
    - For source_urls, include only valid URLs that appear in the answer. If none are mentioned, return an empty array.
    - Keep the original phrasing of cities/locations from the answer.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _safe_join_sources(sources: Optional[List[str]]) -> List[str]:
    if not sources:
        return []
    # Filter empty strings and obvious non-URLs
    return [s for s in sources if isinstance(s, str) and s.strip()]


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_pennsylvania_info(
    evaluator: Evaluator,
    parent_node,
    pa: Optional[PennsylvaniaSchoolInfo],
) -> None:
    # Create Pennsylvania node (parallel, non-critical parent)
    pa_node = evaluator.add_parallel(
        id="Pennsylvania_School_Information",
        desc="Information about the Big Ten Conference school with its main campus in Pennsylvania",
        parent=parent_node,
        critical=False,
    )

    school_name_val = pa.school_name if pa else None
    campus_loc_val = pa.campus_location if pa else None
    ad_name_val = pa.athletic_director_name if pa else None
    ad_start_year_val = pa.athletic_director_start_year if pa else None
    sources = _safe_join_sources(pa.source_urls if pa else [])

    # 1) School_Name (critical)
    school_name_leaf = evaluator.add_leaf(
        id="PA_School_Name",
        desc="Correctly identifies the name of the Big Ten Conference school with its main campus in Pennsylvania",
        parent=pa_node,
        critical=True,
    )
    school_claim = (
        f"The Big Ten school whose main campus is in Pennsylvania is {school_name_val}."
        if school_name_val else "No valid Pennsylvania Big Ten school name was provided."
    )
    await evaluator.verify(
        claim=school_claim,
        node=school_name_leaf,
        sources=sources,
        additional_instruction="Use the provided sources to confirm the school's Big Ten membership and that its main campus is in Pennsylvania.",
    )

    # 2) Campus_Location (critical)
    campus_leaf = evaluator.add_leaf(
        id="PA_Campus_Location",
        desc="Correctly specifies the city or cities where this Pennsylvania school's main campus is located",
        parent=pa_node,
        critical=True,
    )
    campus_claim = (
        f"The main campus of {school_name_val} is located in {campus_loc_val}."
        if school_name_val and campus_loc_val else "The main campus location for the Pennsylvania school was not clearly provided."
    )
    await evaluator.verify(
        claim=campus_claim,
        node=campus_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm the official main campus city name(s). For Penn State, 'University Park' and 'State College' are often used; treat them reasonably as equivalent if sources indicate."
        ),
    )

    # 3) Athletic_Director_Name (critical)
    ad_name_leaf = evaluator.add_leaf(
        id="PA_Athletic_Director_Name",
        desc="Correctly provides the full name of this Pennsylvania school's current athletic director as of 2024",
        parent=pa_node,
        critical=True,
    )
    ad_name_claim = (
        f"The current athletic director of {school_name_val} (as of the 2024-2025 season) is {ad_name_val}."
        if school_name_val and ad_name_val else "The Pennsylvania school's current athletic director name was not clearly provided."
    )
    await evaluator.verify(
        claim=ad_name_claim,
        node=ad_name_leaf,
        sources=sources,
        additional_instruction="Confirm the individual's role as Director of Athletics/AD on official sources or reputable pages; treat equivalent titles reasonably.",
    )

    # 4) Athletic_Director_Start_Year (critical)
    ad_year_leaf = evaluator.add_leaf(
        id="PA_Athletic_Director_Start_Year",
        desc="Correctly states the year this person began serving as athletic director at this school",
        parent=pa_node,
        critical=True,
    )
    ad_year_claim = (
        f"{ad_name_val} began serving as athletic director at {school_name_val} in {ad_start_year_val}."
        if school_name_val and ad_name_val and ad_start_year_val else "The start year for the Pennsylvania school's athletic director was not clearly provided."
    )
    await evaluator.verify(
        claim=ad_year_claim,
        node=ad_year_leaf,
        sources=sources,
        additional_instruction="Verify the appointment/start year; if a full date is shown, ensure the year component matches.",
    )


async def verify_new_jersey_info(
    evaluator: Evaluator,
    parent_node,
    nj: Optional[NewJerseySchoolInfo],
) -> None:
    # Create New Jersey node (parallel, non-critical parent)
    nj_node = evaluator.add_parallel(
        id="New_Jersey_School_Information",
        desc="Information about the Big Ten Conference school with its main campus in New Jersey",
        parent=parent_node,
        critical=False,
    )

    school_name_val = nj.school_name if nj else None
    campus_loc_val = nj.campus_location if nj else None
    ad_status_val = (nj.athletic_director_status or "").strip().lower() if nj and nj.athletic_director_status else None
    interim_name_val = nj.interim_athletic_director_name if nj else None
    sources = _safe_join_sources(nj.source_urls if nj else [])

    # 1) School_Name (critical)
    school_name_leaf = evaluator.add_leaf(
        id="NJ_School_Name",
        desc="Correctly identifies the name of the Big Ten Conference school with its main campus in New Jersey",
        parent=nj_node,
        critical=True,
    )
    school_claim = (
        f"The Big Ten school whose main campus is in New Jersey is {school_name_val}."
        if school_name_val else "No valid New Jersey Big Ten school name was provided."
    )
    await evaluator.verify(
        claim=school_claim,
        node=school_name_leaf,
        sources=sources,
        additional_instruction="Use the provided sources to confirm the school's Big Ten membership and that its main campus is in New Jersey.",
    )

    # 2) Campus_Location (critical)
    campus_leaf = evaluator.add_leaf(
        id="NJ_Campus_Location",
        desc="Correctly specifies the city or cities where this New Jersey school's main campus is located",
        parent=nj_node,
        critical=True,
    )
    campus_claim = (
        f"The main campus of {school_name_val} is located in {campus_loc_val}."
        if school_name_val and campus_loc_val else "The main campus location for the New Jersey school was not clearly provided."
    )
    await evaluator.verify(
        claim=campus_claim,
        node=campus_leaf,
        sources=sources,
        additional_instruction="Confirm official main campus city name(s); for Rutgers, New Brunswick and/or Piscataway are commonly cited.",
    )

    # 3) Athletic_Director_Status (critical)
    ad_status_leaf = evaluator.add_leaf(
        id="NJ_Athletic_Director_Status",
        desc="Correctly indicates whether this New Jersey school currently has a permanent or interim athletic director as of 2024",
        parent=nj_node,
        critical=True,
    )
    if ad_status_val in ("permanent", "interim"):
        ad_status_claim = f"As of the 2024-2025 season, {school_name_val} currently has a {ad_status_val} athletic director."
    else:
        ad_status_claim = "The current athletic director status (permanent or interim) for the New Jersey school was not clearly provided."
    await evaluator.verify(
        claim=ad_status_claim,
        node=ad_status_leaf,
        sources=sources,
        additional_instruction="Verify whether the AD role is permanent or interim from the sources; treat equivalent phrasing reasonably.",
    )

    # 4) Interim_Athletic_Director_Name (non-critical; only applicable if interim)
    if ad_status_val == "interim":
        interim_leaf = evaluator.add_leaf(
            id="NJ_Interim_Athletic_Director_Name",
            desc="If the New Jersey school's athletic director position is interim, correctly provides the interim athletic director's full name",
            parent=nj_node,
            critical=False,  # Adjusted criticality: only applicable if interim; non-critical otherwise
        )
        interim_claim = (
            f"The interim athletic director of {school_name_val} is {interim_name_val}."
            if school_name_val and interim_name_val else "The interim athletic director's name was not clearly provided for the New Jersey school."
        )
        await evaluator.verify(
            claim=interim_claim,
            node=interim_leaf,
            sources=sources,
            additional_instruction="Confirm the interim AD's full name on official athletics or school press releases.",
        )
    else:
        # Not applicable; add a skipped leaf to reflect rubric item but mark as not applicable
        evaluator.add_leaf(
            id="NJ_Interim_Athletic_Director_Name",
            desc="If the New Jersey school's athletic director position is interim, correctly provides the interim athletic director's full name",
            parent=nj_node,
            critical=False,
            score=0.0,
            status="skipped",
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
    Evaluate an answer for Big Ten PA/NJ schools and AD information task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates PA and NJ info independently
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

    # Extract structured PA/NJ information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_pa_nj(),
        template_class=PA_NJ_Extraction,
        extraction_name="pa_nj_extraction",
    )

    # Build top-level rubric node (optional child under root to mirror rubric name)
    # Note: The evaluator root already contains TASK_DESCRIPTION; we add a parallel child to reflect rubric structure.
    rubric_node = evaluator.add_parallel(
        id="Big_Ten_PA_NJ_Schools_Information",
        desc="Identifies Big Ten Conference schools with main campuses in Pennsylvania or New Jersey for the 2024-2025 season and provides required information about each school's location and athletic director",
        parent=root,
        critical=False,
    )

    # Verify Pennsylvania information
    await verify_pennsylvania_info(evaluator, rubric_node, extracted.pennsylvania)

    # Verify New Jersey information
    await verify_new_jersey_info(evaluator, rubric_node, extracted.new_jersey)

    # Return evaluation summary
    return evaluator.get_summary()