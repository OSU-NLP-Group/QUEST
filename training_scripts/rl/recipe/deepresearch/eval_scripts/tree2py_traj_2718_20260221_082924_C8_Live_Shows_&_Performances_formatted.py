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
TASK_ID = "charlotte_broadway_2026_feb_mar"
TASK_DESCRIPTION = """
I'm planning to attend Broadway shows in Charlotte, North Carolina during February or March 2026. Identify 3 touring Broadway productions that will be performing in Charlotte during this timeframe. For each show, provide: (1) the show name, (2) the specific venue name where it will perform, (3) the venue's seating capacity, (4) the exact performance dates, and (5) an official URL from either the show's tour website or the venue's official website that confirms this information.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ShowInfo(BaseModel):
    show_name: Optional[str] = None
    venue_name: Optional[str] = None
    venue_capacity: Optional[str] = None
    performance_dates: Optional[str] = None
    official_url: Optional[str] = None


class CharlotteBroadwayExtraction(BaseModel):
    shows: List[ShowInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_charlotte_broadway() -> str:
    return """
    From the answer, extract up to three distinct touring Broadway productions that are scheduled to perform in Charlotte, North Carolina during February or March 2026.

    For each show, extract the following fields exactly as stated in the answer:
    1. show_name: The production title (musical or play).
    2. venue_name: The specific Charlotte venue where the performance will occur (e.g., Belk Theater, Ovens Auditorium, Knight Theater).
    3. venue_capacity: The seating capacity value stated for the venue; if given as a range or approximate, extract the exact string (do not convert to a number).
    4. performance_dates: The exact performance dates for Charlotte (e.g., "Feb 24–Mar 1, 2026" or specific dates listed). Keep the original formatting and wording from the answer.
    5. official_url: A single official URL that confirms this information. This should be either:
       - the show's official tour schedule page entry for Charlotte, or
       - the venue's official website page listing the show and dates.
       If multiple URLs are present, choose the most official one that best confirms the Charlotte stop and dates. Extract the actual URL string (full URL). If no URL is present, set to null.

    IMPORTANT:
    - Only include shows explicitly mentioned in the answer and that are stated to occur in Charlotte during February or March 2026.
    - If a field is missing for a show, set it to null.
    - If the answer lists more than three shows, extract all and the evaluator will later select the first three.
    - Do not invent or infer any information not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_str(value: Optional[str]) -> str:
    return value or ""


# --------------------------------------------------------------------------- #
# Verification per show                                                       #
# --------------------------------------------------------------------------- #
async def verify_show(
    evaluator: Evaluator,
    parent_node,
    show: ShowInfo,
    idx: int,
    prior_names: List[str],
) -> None:
    """
    Build verification nodes for a single show and run checks.
    The show-level node is non-critical to allow partial credit across shows,
    while each leaf inside is critical to ensure all required facts are supported.
    """

    show_num = idx + 1
    show_node = evaluator.add_parallel(
        id=f"show_{show_num}",
        desc="{} touring Broadway show identified".format(
            ["First", "Second", "Third"][idx] if idx < 3 else f"Show #{show_num}"
        ),
        parent=parent_node,
        critical=False,
    )

    name = _safe_str(show.show_name)
    venue = _safe_str(show.venue_name)
    capacity = _safe_str(show.venue_capacity)
    dates = _safe_str(show.performance_dates)
    url = show.official_url

    # Leaf 1: Official URL provided (presence + officialness judged from the answer text)
    ref_leaf = evaluator.add_leaf(
        id=f"show_{show_num}_reference",
        desc="Official URL from tour website or venue website is provided",
        parent=show_node,
        critical=True,
    )
    claim_ref = (
        f"The answer provides an official URL for the show '{name}' that comes from either the show's tour website "
        f"or the venue's official website. The provided URL is: {url if url else 'None'}."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=ref_leaf,
        sources=None,  # This check is about presence/officialness in the answer context
        additional_instruction="Judge based on the answer text: is there a URL present and does it appear to be official (tour site or venue official domain)?",
    )

    # Leaf 2: Identification — confirmed touring Broadway production scheduled in Charlotte, distinct from prior
    ident_leaf = evaluator.add_leaf(
        id=f"show_{show_num}_identification",
        desc="Show name is provided and is a confirmed touring Broadway production",
        parent=show_node,
        critical=True,
    )
    prior_list_str = ", ".join(prior_names) if prior_names else "none"
    claim_ident = (
        f"The official page confirms that the production '{name}' is a touring Broadway show and that it is scheduled "
        f"to perform in Charlotte, NC. Also confirm that '{name}' is different from previously listed shows: {prior_list_str}."
    )
    await evaluator.verify(
        claim=claim_ident,
        node=ident_leaf,
        sources=url,
        additional_instruction="Verify the page shows a national tour or touring engagement and includes a Charlotte, NC stop for the named show. Allow minor naming variations.",
        extra_prerequisites=[ref_leaf],
    )

    # Leaf 3: Venue — specific venue in Charlotte
    venue_leaf = evaluator.add_leaf(
        id=f"show_{show_num}_venue",
        desc="Specific venue name in Charlotte, NC is provided",
        parent=show_node,
        critical=True,
    )
    claim_venue = (
        f"The official page confirms that the show '{name}' will perform at the venue '{venue}' in Charlotte, North Carolina."
    )
    await evaluator.verify(
        claim=claim_venue,
        node=venue_leaf,
        sources=url,
        additional_instruction="Check the Charlotte entry/listing for the venue name. Accept reasonable formatting variants or appended branding.",
        extra_prerequisites=[ref_leaf],
    )

    # Leaf 4: Dates — performance dates during February or March 2026, with exact dates as stated
    dates_leaf = evaluator.add_leaf(
        id=f"show_{show_num}_dates",
        desc="Performance dates during February or March 2026 are provided",
        parent=show_node,
        critical=True,
    )
    claim_dates = (
        f"The official page lists the Charlotte performance dates exactly as: '{dates}', and these dates occur in February or March 2026."
    )
    await evaluator.verify(
        claim=claim_dates,
        node=dates_leaf,
        sources=url,
        additional_instruction=(
            "Confirm the Charlotte schedule dates match the stated string and fall in Feb or Mar 2026. "
            "If a range crosses months (e.g., Feb 28–Mar 3, 2026), it still qualifies."
        ),
        extra_prerequisites=[ref_leaf],
    )

    # Leaf 5: Capacity — venue seating capacity
    capacity_leaf = evaluator.add_leaf(
        id=f"show_{show_num}_capacity",
        desc="Venue seating capacity is provided",
        parent=show_node,
        critical=True,
    )
    claim_capacity = (
        f"The official page confirms that the seating capacity of the venue '{venue}' is '{capacity}'."
    )
    await evaluator.verify(
        claim=claim_capacity,
        node=capacity_leaf,
        sources=url,
        additional_instruction=(
            "Verify that the provided capacity value is explicitly supported by this official URL. "
            "If capacity is not stated on the page, conclude not supported."
        ),
        extra_prerequisites=[ref_leaf],
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
    Evaluate an answer for the Charlotte Broadway shows (Feb/Mar 2026) task.
    """

    evaluator = Evaluator()
    # Root should be non-critical to allow partial credit across multiple shows
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_charlotte_broadway(),
        template_class=CharlotteBroadwayExtraction,
        extraction_name="charlotte_broadway_shows",
    )

    # Select first 3 shows; pad if fewer
    selected_shows: List[ShowInfo] = list(extracted.shows[:3])
    while len(selected_shows) < 3:
        selected_shows.append(ShowInfo())

    # Verify each show
    prior_names: List[str] = []
    for i, show in enumerate(selected_shows):
        await verify_show(evaluator, root, show, i, prior_names)
        if show.show_name:
            prior_names.append(show.show_name)

    # Return summary
    return evaluator.get_summary()