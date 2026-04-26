import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "closest_open_state_park_wi_dells_2026"
TASK_DESCRIPTION = """
Identify the state park closest to Wisconsin Dells, Wisconsin, that is open for day-use visits on Martin Luther King Jr. Day 2026 (January 19, 2026). Provide the exact distance from Wisconsin Dells to this state park and the park's total acreage.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StateParkExtraction(BaseModel):
    """
    Structured information extracted from the agent's answer.
    """
    park_name: Optional[str] = None
    distance: Optional[str] = None  # Keep as string to allow “about X miles” etc.
    acreage: Optional[str] = None   # Keep as string to allow formatting variations

    # Source URLs explicitly cited in the answer
    open_status_urls: List[str] = Field(default_factory=list)          # URLs supporting open/operating hours on the date
    distance_urls: List[str] = Field(default_factory=list)             # URLs supporting the distance figure
    acreage_urls: List[str] = Field(default_factory=list)              # URLs supporting the total acreage
    closest_comparison_urls: List[str] = Field(default_factory=list)   # URLs supporting the “closest” claim (e.g., comparisons, lists, or closures of nearer parks)
    general_urls: List[str] = Field(default_factory=list)              # Any general/official URLs about the identified park (e.g., DNR page)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_state_park() -> str:
    return """
    Extract the single state park the answer claims is the closest to Wisconsin Dells, Wisconsin AND is open for day-use on Martin Luther King Jr. Day 2026 (January 19, 2026). Also extract the distance and acreage exactly as stated in the answer, and categorize any cited URLs by what they support.

    Return a JSON object with the following fields:
    - park_name: string | null
    - distance: string | null        # the distance from Wisconsin Dells to the identified park, exactly as written in the answer
    - acreage: string | null         # the park's total acreage, exactly as written in the answer

    - open_status_urls: string[]     # URLs that support that this park is open for day-use on Jan 19, 2026 (or generally open year-round/day-use hours)
    - distance_urls: string[]        # URLs that support the stated distance from Wisconsin Dells to the park (e.g., Google Maps directions)
    - acreage_urls: string[]         # URLs that support the total acreage figure
    - closest_comparison_urls: string[]  # URLs that support the “closest open state park” claim (e.g., comparisons, lists of distances, or evidence that any nearer parks are closed that date)
    - general_urls: string[]         # any other URLs cited about the identified park (official park page, DNR page, etc.)

    IMPORTANT:
    - Only include URLs explicitly present in the answer (plain URL or markdown link target). Do not invent or infer URLs.
    - If a field is not present in the answer, set it to null (for strings) or [] (for URL lists).
    - Preserve the distance and acreage text exactly as written (including units or approximations).
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _combine_unique_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            u_norm = (u or "").strip()
            if not u_norm:
                continue
            if u_norm not in seen:
                seen.add(u_norm)
                combined.append(u_norm)
    return combined


# --------------------------------------------------------------------------- #
# Verification sub-tree                                                       #
# --------------------------------------------------------------------------- #
async def verify_state_park_information(
    evaluator: Evaluator,
    parent_node,
    extracted: StateParkExtraction
) -> None:
    """
    Build and verify the rubric tree for:
    - Correct State Park (closest open on MLK Day 2026)
    - Distance accuracy
    - Acreage information
    """
    # Top-level critical node (parallel aggregation)
    spi_node = evaluator.add_parallel(
        id="State_Park_Information",
        desc="Identification and factual details about the state park closest to Wisconsin Dells that is open for day-use visits on Martin Luther King Jr. Day 2026 (January 19, 2026)",
        parent=parent_node,
        critical=True
    )

    # -------------------- Correct State Park (expand into sub-checks) --------------------
    correct_sp_node = evaluator.add_parallel(
        id="Correct_State_Park",
        desc="The state park identified is the closest one to Wisconsin Dells among nearby state parks and is confirmed to be open for day-use visits on January 19, 2026",
        parent=spi_node,
        critical=True
    )

    # Existence check for park name (critical)
    evaluator.add_custom_node(
        result=bool(extracted.park_name and extracted.park_name.strip()),
        id="park_name_provided",
        desc="A specific state park name is provided",
        parent=correct_sp_node,
        critical=True
    )

    # Open on MLK Day 2026 (critical)
    open_leaf = evaluator.add_leaf(
        id="open_on_mlk_day_2026",
        desc="The identified park is open for day-use on January 19, 2026 (Martin Luther King Jr. Day)",
        parent=correct_sp_node,
        critical=True
    )
    open_claim = (
        f"{extracted.park_name or 'The park'} is open for day-use visits on January 19, 2026 (Martin Luther King Jr. Day)."
    )
    open_sources = _combine_unique_urls(extracted.open_status_urls, extracted.general_urls)
    await evaluator.verify(
        claim=open_claim,
        node=open_leaf,
        sources=open_sources,
        additional_instruction=(
            "Focus on day-use access, not camping or facilities. A page stating the park is open year-round or "
            "lists daily day-use hours (e.g., 6 a.m.–11 p.m.) counts as support for being open on Jan 19, 2026, "
            "unless the page explicitly lists a closure on that specific date."
        ),
    )

    # Closest open state park to Wisconsin Dells (critical)
    closest_leaf = evaluator.add_leaf(
        id="closest_open_state_park",
        desc="The identified park is the closest state park to Wisconsin Dells that is open for day-use on the specified date",
        parent=correct_sp_node,
        critical=True
    )
    closest_claim = (
        f"Among Wisconsin state parks, {extracted.park_name or 'the identified park'} is the closest state park to Wisconsin Dells, Wisconsin, "
        f"that is open for day-use on January 19, 2026."
    )
    # Try all potentially relevant sources provided by the answer for this comparative claim
    closest_sources = _combine_unique_urls(
        extracted.closest_comparison_urls,
        extracted.distance_urls,
        extracted.open_status_urls,
        extracted.general_urls
    )
    await evaluator.verify(
        claim=closest_claim,
        node=closest_leaf,
        sources=closest_sources,
        additional_instruction=(
            "This is a comparative claim. To pass, the single checked page must explicitly support the conclusion that "
            "the named park is the closest open state park to Wisconsin Dells on the specified date. This can be via an "
            "explicit statement or a clear table/list that shows both (1) distances to multiple parks from Wisconsin Dells "
            "and (2) that any nearer parks are closed that date. If the page only shows a distance for the chosen park "
            "without establishing that nearer parks are closed or farther away, mark as not supported."
        ),
    )

    # -------------------- Distance Accuracy (critical) --------------------
    distance_leaf = evaluator.add_leaf(
        id="Distance_Accuracy",
        desc="The distance from Wisconsin Dells to the identified state park is accurately provided",
        parent=spi_node,
        critical=True
    )
    distance_claim = (
        f"The distance from Wisconsin Dells, Wisconsin to {extracted.park_name or 'the identified park'} is {extracted.distance or 'N/A'}."
    )
    await evaluator.verify(
        claim=distance_claim,
        node=distance_leaf,
        sources=_combine_unique_urls(extracted.distance_urls, extracted.closest_comparison_urls),
        additional_instruction=(
            "Verify the stated distance (road or straight-line if clearly indicated). Allow minor rounding or route variations "
            "(±10% is acceptable). Prefer distances measured from the city center of Wisconsin Dells to the park's primary entrance."
        ),
    )

    # -------------------- Acreage Information (critical) --------------------
    acreage_leaf = evaluator.add_leaf(
        id="Acreage_Information",
        desc="The total acreage of the identified state park is provided",
        parent=spi_node,
        critical=True
    )
    acreage_claim = (
        f"The total acreage of {extracted.park_name or 'the identified park'} is {extracted.acreage or 'N/A'}."
    )
    await evaluator.verify(
        claim=acreage_claim,
        node=acreage_leaf,
        sources=_combine_unique_urls(extracted.acreage_urls, extracted.general_urls),
        additional_instruction=(
            "Confirm the park's total acreage as stated. Accept slight rounding differences and formatting (e.g., commas). "
            "Ensure the figure refers to the entire state park, not just a lake surface area or a sub-area."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 'closest open state park to Wisconsin Dells on MLK Day 2026' task.
    """
    # Initialize evaluator
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_state_park(),
        template_class=StateParkExtraction,
        extraction_name="state_park_extraction",
    )

    # Build and verify tree according to rubric
    await verify_state_park_information(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()