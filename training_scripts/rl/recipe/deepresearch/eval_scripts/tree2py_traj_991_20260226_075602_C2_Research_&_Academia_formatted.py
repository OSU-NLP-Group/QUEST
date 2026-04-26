import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "astro_planning_2026"
TASK_DESCRIPTION = (
    "An astrophotography researcher is planning a field observation trip in 2026 to capture both the northern lights "
    "(aurora borealis) and a total lunar eclipse during the same expedition. To maximize their chances of successful "
    "aurora observation, they want to schedule the trip during an equinox period, which is statistically optimal for "
    "geomagnetic storm activity. Additionally, they need a location that is both within the auroral oval for reliable "
    "aurora viewing and offers visibility for the 2026 total lunar eclipse. Identify which month in 2026 provides this "
    "opportunity, and recommend a specific location (city, state/province, or region) that satisfies both requirements. "
    "Provide supporting reference URLs for your answer."
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class PlanningExtraction(BaseModel):
    """
    Structured extraction for the astrophotography planning answer.
    """
    month: Optional[str] = None
    eclipse_date: Optional[str] = None
    location_name: Optional[str] = None

    # Source URLs categorized by purpose
    eclipse_urls: List[str] = Field(default_factory=list)
    aurora_urls: List[str] = Field(default_factory=list)
    equinox_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_planning_info() -> str:
    return """
    Extract the key planning details for the 2026 aurora + total lunar eclipse expedition from the answer.

    Return a JSON object containing these fields:
    1) month: The month in 2026 that the answer recommends scheduling the trip (e.g., "March 2026" or "March").
    2) eclipse_date: The specific date of the total lunar eclipse as stated in the answer (e.g., "March 3, 2026"). If not explicitly stated, return null.
    3) location_name: The recommended observing location (city, state/province, or region). If multiple are listed, choose the primary recommendation; otherwise return the first one mentioned. If none, return null.

    4) eclipse_urls: All URLs cited that directly support the eclipse timing or visibility region for the 2026 total lunar eclipse.
    5) aurora_urls: All URLs cited that support aurora viewing reliability or confirm the location is within/near the auroral oval.
    6) equinox_urls: All URLs cited that support the claim that equinox periods (around March 20–21 or September 22–23) are statistically optimal for aurora activity.

    Special rules for URL extraction:
    - Extract only actual URLs present in the answer (plain or markdown links). Do not invent any URLs.
    - Include full URLs with protocol (http:// or https://). If a URL is missing protocol, prepend http://.
    - If no URLs are provided for a category, return an empty array for that category.

    If any of the above fields are missing in the answer, set them to null (or empty arrays for URL lists).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_optimal_month_node(
    evaluator: Evaluator,
    parent_node,
    data: PlanningExtraction
) -> None:
    """
    Build and verify the 'Optimal_Month_Identification' subtree.
    """
    node = evaluator.add_parallel(
        id="Optimal_Month_Identification",
        desc="Identify March 2026 as the month that coincides with both a total lunar eclipse and an equinox period for optimal aurora viewing",
        parent=parent_node,
        critical=False
    )

    # Optional existence gate to ensure the answer provided a month
    month_provided = evaluator.add_custom_node(
        result=bool(data.month and data.month.strip()),
        id="Month_Provided",
        desc="Answer provides a month recommendation for 2026",
        parent=node,
        critical=True
    )

    # Eclipse_Month leaf
    eclipse_month_leaf = evaluator.add_leaf(
        id="Eclipse_Month",
        desc="Correctly identify that the total lunar eclipse occurs in March 2026 (specifically March 3, 2026)",
        parent=node,
        critical=True
    )

    # Claim uses the answer's stated month and date if available; sources must support it
    stated_month = data.month or "March 2026"
    stated_date = data.eclipse_date or "March 3, 2026"
    eclipse_claim = f"A total lunar eclipse occurs in {stated_month}, specifically on {stated_date}."

    await evaluator.verify(
        claim=eclipse_claim,
        node=eclipse_month_leaf,
        sources=data.eclipse_urls,
        additional_instruction=(
            "Verify via the provided eclipse references (e.g., NASA/Timeanddate) that the 2026 total lunar eclipse occurs "
            "in March, specifically around March 3, 2026. Minor timezone or slight wording variations are acceptable."
        )
    )

    # Equinox_Proximity leaf
    equinox_leaf = evaluator.add_leaf(
        id="Equinox_Proximity",
        desc="Verify that March is near the spring equinox (around March 21), which is statistically optimal for aurora viewing",
        parent=node,
        critical=True
    )

    # Use equinox URLs; if absent, fall back to aurora URLs that discuss equinox-related auroral activity
    equinox_sources = data.equinox_urls if data.equinox_urls else data.aurora_urls
    equinox_claim = (
        f"The month '{stated_month}' aligns with the spring equinox period (around March 20–21), "
        "and equinoxes are known to statistically enhance aurora activity."
    )

    await evaluator.verify(
        claim=equinox_claim,
        node=equinox_leaf,
        sources=equinox_sources,
        additional_instruction=(
            "Confirm that equinox periods (around March 20–21) are associated with increased geomagnetic activity and higher aurora "
            "occurrence rates. The sources should explicitly or clearly support the equinox-aurora connection."
        )
    )


async def build_location_selection_node(
    evaluator: Evaluator,
    parent_node,
    data: PlanningExtraction
) -> None:
    """
    Build and verify the 'Location_Selection' subtree.
    """
    node = evaluator.add_parallel(
        id="Location_Selection",
        desc="Identify an appropriate location within the auroral oval that also has visibility for the March 2026 total lunar eclipse",
        parent=parent_node,
        critical=False
    )

    # Existence gate to ensure the answer provided a location
    location_provided = evaluator.add_custom_node(
        result=bool(data.location_name and data.location_name.strip()),
        id="Location_Provided",
        desc="Answer provides a specific observing location",
        parent=node,
        critical=True
    )

    # Auroral_Oval_Membership leaf
    oval_leaf = evaluator.add_leaf(
        id="Auroral_Oval_Membership",
        desc="The selected location must be within or near the auroral oval (e.g., Alaska, Canada, northern Scandinavia, northern Iceland, Greenland)",
        parent=node,
        critical=True
    )

    oval_claim = (
        f"The selected location '{data.location_name}' is within or near the auroral oval and is a reliable aurora viewing location."
        if data.location_name else
        "The selected location is within or near the auroral oval and is a reliable aurora viewing location."
    )

    await evaluator.verify(
        claim=oval_claim,
        node=oval_leaf,
        sources=data.aurora_urls,
        additional_instruction=(
            "Use the provided aurora references (e.g., NOAA, Geophysical Institute, national observatory pages) to confirm "
            "that the location lies within or near the auroral oval or is widely recognized as a reliable aurora viewing site."
        )
    )

    # Eclipse_Visibility leaf
    visibility_leaf = evaluator.add_leaf(
        id="Eclipse_Visibility",
        desc="The selected location must be in a region with visibility for the March 3, 2026 total lunar eclipse (western North America, Australia, Pacific, or East Asia)",
        parent=node,
        critical=True
    )

    visibility_claim = (
        f"The selected location '{data.location_name}' is in a region that will have visibility of the total lunar eclipse on March 3, 2026."
        if data.location_name else
        "The selected location is in a region that will have visibility of the total lunar eclipse on March 3, 2026."
    )

    await evaluator.verify(
        claim=visibility_claim,
        node=visibility_leaf,
        sources=data.eclipse_urls,
        additional_instruction=(
            "From the provided eclipse references, verify that the location falls within the stated visibility regions "
            "(e.g., western North America, Australia, Pacific, East Asia) for the March 3, 2026 total lunar eclipse. "
            "If the source lists regions rather than cities, reason whether the location belongs to one of those regions."
        )
    )


async def build_reference_urls_node(
    evaluator: Evaluator,
    parent_node,
    data: PlanningExtraction
) -> None:
    """
    Build the 'Reference_URLs' node to check that supporting URLs were provided.
    """
    # Presence check: must include eclipse URLs and at least one aurora/equinox URL
    has_refs = (len(data.eclipse_urls) > 0) and ((len(data.aurora_urls) > 0) or (len(data.equinox_urls) > 0))

    evaluator.add_custom_node(
        result=has_refs,
        id="Reference_URLs",
        desc="Provide supporting reference URLs that validate the month and location recommendations",
        parent=parent_node,
        critical=False
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
    Evaluate an answer for the 2026 aurora + total lunar eclipse planning task.
    """
    # Initialize evaluator; root must be non-critical to allow partial credit aggregation
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

    # Add top-level planning node (set non-critical to avoid critical consistency constraint)
    planning_root = evaluator.add_parallel(
        id="Astronomical_Observation_Planning",
        desc="Correctly identify the optimal month and location for observing both a total lunar eclipse and northern lights during an equinox period in 2026, with supporting references",
        parent=root,
        critical=False
    )

    # Extract structured information from the answer
    extracted: PlanningExtraction = await evaluator.extract(
        prompt=prompt_extract_planning_info(),
        template_class=PlanningExtraction,
        extraction_name="planning_extraction"
    )

    # Build and verify subtrees
    await build_optimal_month_node(evaluator, planning_root, extracted)
    await build_location_selection_node(evaluator, planning_root, extracted)
    await build_reference_urls_node(evaluator, planning_root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()