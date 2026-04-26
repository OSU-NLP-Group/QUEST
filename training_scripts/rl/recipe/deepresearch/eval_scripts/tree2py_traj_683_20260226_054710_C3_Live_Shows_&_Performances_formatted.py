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
TASK_ID = "manhattan_nhl_venue_1962_capacity_2790"
TASK_DESCRIPTION = (
    "Identify the performance venue in Manhattan that was designated as a National Historic Landmark in 1962 "
    "and has a main auditorium with a seating capacity of 2,790 seats. Provide the venue's full name and the year "
    "it was subsequently designated as a New York City Landmark."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueSources(BaseModel):
    """Categorized URLs cited in the answer that support specific claims."""
    location: List[str] = Field(default_factory=list)
    venue_type: List[str] = Field(default_factory=list)
    nhl: List[str] = Field(default_factory=list)
    capacity: List[str] = Field(default_factory=list)
    nyc_landmark: List[str] = Field(default_factory=list)
    general: List[str] = Field(default_factory=list)


class VenueExtraction(BaseModel):
    """Structured extraction from the agent's answer."""
    venue_name: Optional[str] = None
    nhl_year_stated_in_answer: Optional[str] = None
    main_auditorium_capacity_stated_in_answer: Optional[str] = None
    nyc_landmark_year: Optional[str] = None
    sources: VenueSources = Field(default_factory=VenueSources)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    Extract the single performance venue identified in the answer that is claimed to be:
    - located in Manhattan, New York City;
    - designated as a National Historic Landmark in 1962; and
    - having a main auditorium seating capacity of 2,790 seats;
    Also extract the year the venue was designated as a New York City Landmark.

    Return a JSON object with:
    - venue_name: Full official name of the venue (string).
    - nhl_year_stated_in_answer: The year of the National Historic Landmark designation explicitly stated (string; if multiple years mentioned, choose the one tied to NHL designation; if unspecified, null).
    - main_auditorium_capacity_stated_in_answer: The capacity number explicitly stated for the main auditorium (string; keep formatting such as "2,790" if present; if unspecified, null).
    - nyc_landmark_year: The year the venue was designated a New York City Landmark (string; if unspecified, null).
    - sources: Categorize all URLs explicitly mentioned in the answer into the following arrays:
        • location: URLs supporting the Manhattan location claim.
        • venue_type: URLs supporting that it is a performance venue (concert hall, theater, etc.).
        • nhl: URLs supporting the National Historic Landmark designation in 1962.
        • capacity: URLs supporting the main auditorium seating capacity claim.
        • nyc_landmark: URLs supporting the NYC Landmark designation year.
        • general: Any other URLs cited in the answer that are relevant but not clearly categorized.

    Rules:
    - Extract only information explicitly present in the answer.
    - For each URL field, extract actual URLs (plain or markdown links). If none provided, return an empty list.
    - Do not invent URLs or facts; if a field is missing in the answer, return null (for strings) or [] (for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def dedup_merge(*lists: List[str]) -> List[str]:
    """Merge multiple URL lists and deduplicate while preserving order."""
    seen = set()
    merged = []
    for lst in lists:
        for url in lst or []:
            if url and url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


def safe_str(x: Optional[str]) -> str:
    return (x or "").strip()


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_venue(
    evaluator: Evaluator,
    parent_node,
    info: VenueExtraction,
) -> None:
    """
    Build the verification tree and run all checks according to the rubric.
    """

    # 1) Venue Identification block (Sequential, Critical)
    venue_ident_node = evaluator.add_sequential(
        id="Venue_Identification",
        desc="A specific performance venue name is provided",
        parent=parent_node,
        critical=True
    )

    # 1.1) Name provided (existence check) - Critical leaf
    name_provided = bool(safe_str(info.venue_name))
    evaluator.add_custom_node(
        result=name_provided,
        id="Venue_Name_Provided",
        desc="A specific venue name is present in the answer",
        parent=venue_ident_node,
        critical=True
    )

    # 2) Attributes Verification (Parallel, Critical)
    attrs_node = evaluator.add_parallel(
        id="Venue_Attributes_Verification",
        desc="The identified venue's attributes are verified against the required criteria",
        parent=venue_ident_node,
        critical=True
    )

    venue_name = safe_str(info.venue_name)

    # 2.1) Manhattan Location - Critical leaf (verify by URLs)
    manhattan_node = evaluator.add_leaf(
        id="Manhattan_Location",
        desc="The identified venue is located in Manhattan, New York City",
        parent=attrs_node,
        critical=True
    )
    manhattan_sources = dedup_merge(info.sources.location, info.sources.general)
    manhattan_claim = f"The venue '{venue_name}' is located in Manhattan, New York City."
    await evaluator.verify(
        claim=manhattan_claim,
        node=manhattan_node,
        sources=manhattan_sources,
        additional_instruction=(
            "Confirm the venue address is in Manhattan (borough of New York City). "
            "Addresses such as 'New York, NY 100xx' with streets in Manhattan or references to Midtown/Upper West/Upper East, "
            "or explicit mention of Manhattan should be treated as supporting evidence."
        ),
    )

    # 2.2) Performance Venue Type - Critical leaf (verify by URLs)
    perf_type_node = evaluator.add_leaf(
        id="Performance_Venue_Type",
        desc="The venue is a performance venue (concert hall, theater, or similar cultural institution)",
        parent=attrs_node,
        critical=True
    )
    perf_type_sources = dedup_merge(info.sources.venue_type, info.sources.general)
    perf_type_claim = (
        f"The venue '{venue_name}' is a performance venue such as a concert hall, theater, opera house, "
        f"or similar cultural institution where performances are held."
    )
    await evaluator.verify(
        claim=perf_type_claim,
        node=perf_type_node,
        sources=perf_type_sources,
        additional_instruction=(
            "Verify that the venue is described as a concert hall, theater, opera house, or similar performance space. "
            "Descriptions indicating music, concerts, performances, stage, or auditorium support this claim."
        ),
    )

    # 2.3) National Historic Landmark (1962) - Parallel, Critical group
    nhl_group = evaluator.add_parallel(
        id="National_Historic_Landmark_1962",
        desc="The venue was designated as a National Historic Landmark in 1962",
        parent=attrs_node,
        critical=True
    )

    # 2.3.a) Reference URL exists for NHL - Critical custom node (due to framework constraint on critical parent)
    nhl_sources = dedup_merge(info.sources.nhl, info.sources.general)
    evaluator.add_custom_node(
        result=len(info.sources.nhl) > 0,  # existence specifically in the dedicated field
        id="Reference_URL_NHL",
        desc="A reference URL is provided supporting the National Historic Landmark designation in 1962",
        parent=nhl_group,
        critical=True  # Must be critical because parent is critical (framework constraint)
    )

    # 2.3.b) NHL designation fact (1962) - Critical leaf verified by URLs
    nhl_fact_node = evaluator.add_leaf(
        id="NHL_Designation_Fact",
        desc="The National Historic Landmark designation in 1962 is stated",
        parent=nhl_group,
        critical=True
    )
    nhl_claim = f"The venue '{venue_name}' was designated as a National Historic Landmark in 1962."
    await evaluator.verify(
        claim=nhl_claim,
        node=nhl_fact_node,
        sources=nhl_sources,
        additional_instruction=(
            "Confirm that the venue is explicitly listed as a National Historic Landmark (NHL) and that the designation year is 1962. "
            "Do not confuse with the National Register of Historic Places listing; specifically look for NHL status and the year 1962."
        ),
    )

    # 2.4) Main Auditorium Capacity (2,790) - Parallel, Critical group
    capacity_group = evaluator.add_parallel(
        id="Main_Auditorium_Capacity_2790",
        desc="The venue's main auditorium has a seating capacity of 2,790 seats",
        parent=attrs_node,
        critical=True
    )

    # 2.4.a) Reference URL exists for capacity - Critical custom node
    capacity_sources = dedup_merge(info.sources.capacity, info.sources.general)
    evaluator.add_custom_node(
        result=len(info.sources.capacity) > 0,
        id="Reference_URL_Capacity",
        desc="A reference URL is provided supporting the main auditorium seating capacity of 2,790",
        parent=capacity_group,
        critical=True
    )

    # 2.4.b) Capacity fact - Critical leaf verified by URLs
    capacity_fact_node = evaluator.add_leaf(
        id="Capacity_Fact",
        desc="The main auditorium seating capacity of 2,790 is stated",
        parent=capacity_group,
        critical=True
    )
    capacity_claim = f"The main auditorium of '{venue_name}' has a seating capacity of 2,790 seats."
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_fact_node,
        sources=capacity_sources,
        additional_instruction=(
            "Verify the stated capacity specifically for the main auditorium. "
            "If the venue has multiple halls, make sure the number refers to the principal/main auditorium (e.g., 'Stern Auditorium', 'Perelman Stage', 'Main Hall'). "
            "Accept minor phrasing variations like '2,790 seats' or 'seating capacity of 2,790'."
        ),
    )

    # 2.5) NYC Landmark Year Provided - Parallel, Critical group
    nyc_group = evaluator.add_parallel(
        id="NYC_Landmark_Year_Provided",
        desc="The year of New York City Landmark designation is provided",
        parent=attrs_node,
        critical=True
    )

    # Optional: check that a year is provided (existence gating)
    nyc_year_provided = bool(safe_str(info.nyc_landmark_year))
    evaluator.add_custom_node(
        result=nyc_year_provided,
        id="NYC_Landmark_Year_Stated",
        desc="The year of NYC Landmark designation is stated in the answer",
        parent=nyc_group,
        critical=True
    )

    # 2.5.a) Reference URL exists for NYC Landmark - Critical custom node
    nyc_sources = dedup_merge(info.sources.nyc_landmark, info.sources.general)
    evaluator.add_custom_node(
        result=len(info.sources.nyc_landmark) > 0,
        id="Reference_URL_NYC_Landmark",
        desc="A reference URL is provided supporting the New York City Landmark designation year",
        parent=nyc_group,
        critical=True
    )

    # 2.5.b) NYC Landmark year fact - Critical leaf verified by URLs
    nyc_year_fact_node = evaluator.add_leaf(
        id="NYC_Landmark_Year_Fact",
        desc="The year of NYC Landmark designation is stated",
        parent=nyc_group,
        critical=True
    )
    nyc_year_str = safe_str(info.nyc_landmark_year)
    nyc_claim = f"The venue '{venue_name}' was designated as a New York City Landmark in {nyc_year_str}."
    await evaluator.verify(
        claim=nyc_claim,
        node=nyc_year_fact_node,
        sources=nyc_sources,
        additional_instruction=(
            "Verify that the Landmarks Preservation Commission (NYC) designated the venue as a New York City Landmark in the specified year. "
            "If the source provides a full date (e.g., month/day/year), treat the year portion as the key matching criterion."
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
) -> Dict:
    """
    Evaluate an agent's answer for the Manhattan performance venue task using the Mind2Web2 framework.
    """
    # Initialize evaluator with SEQUENTIAL root (aligns with Task_Completion semantics)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    # Build verification tree following the rubric and run checks
    await verify_venue(evaluator, root, extraction)

    # Return structured summary
    return evaluator.get_summary()