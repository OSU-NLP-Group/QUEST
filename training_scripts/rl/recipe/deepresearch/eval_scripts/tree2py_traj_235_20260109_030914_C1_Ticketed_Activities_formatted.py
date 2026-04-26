import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "largest_concert_stadium_ny_nj"
TASK_DESCRIPTION = """
Identify the stadium in the New York/New Jersey metropolitan area that has the largest seating capacity for concerts. Provide the venue name, its concert seating capacity, and a reference URL supporting this information.
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class VenueInfo(BaseModel):
    """
    Information extracted from the answer.
    """
    venue_name: Optional[str] = None
    concert_seating_capacity: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    From the provided answer, extract the following information about the identified stadium:
    1) venue_name: the name of the stadium that is claimed to have the largest concert seating capacity in the New York/New Jersey metropolitan area
    2) concert_seating_capacity: the concert seating capacity figure stated for that stadium (as a string, exactly as written in the answer; include units such as "82,500" if present)
    3) reference_urls: a list of every URL cited to support the claim (extract only valid URLs explicitly present in the answer; include full URLs)
    
    If any field is missing, set it to null (for strings) or an empty list (for reference_urls).
    Only extract one stadium: if multiple are mentioned, choose the primary one the answer presents as the largest concert capacity.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_str(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _safe_name(venue: Optional[str]) -> str:
    return venue if _nonempty_str(venue) else "the identified venue"


# --------------------------------------------------------------------------- #
# Build verification tree and run checks                                      #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    parent_node,
    extracted: VenueInfo,
) -> None:
    """
    Build the rubric tree under a critical top-level node and run all verifications.
    """

    # Create the top-level critical node mirroring the rubric root
    task_node = evaluator.add_parallel(
        id="Largest_Concert_Stadium_NY_NJ",
        desc="Identify the stadium in the New York/New Jersey metropolitan area with the largest concert seating capacity and provide required supporting details",
        parent=parent_node,
        critical=True,
    )

    venue = extracted.venue_name
    capacity = extracted.concert_seating_capacity
    urls = extracted.reference_urls if extracted.reference_urls else []

    # 1) Venue_Name_Provided (existence)
    evaluator.add_custom_node(
        result=_nonempty_str(venue),
        id="Venue_Name_Provided",
        desc="Answer provides the venue/stadium name",
        parent=task_node,
        critical=True,
    )

    # 2) Geographic_Location (verify with sources)
    node_geo = evaluator.add_leaf(
        id="Geographic_Location",
        desc="The identified venue is located in the New York/New Jersey metropolitan area",
        parent=task_node,
        critical=True,
    )
    claim_geo = f"{_safe_name(venue)} is located in the New York/New Jersey metropolitan area (i.e., the New York City metro area that includes parts of New York and New Jersey)."
    await evaluator.verify(
        claim=claim_geo,
        node=node_geo,
        sources=urls,
        additional_instruction="Accept locations such as East Rutherford (NJ), Meadowlands area, or New York City boroughs as inside the NY/NJ metro. Reject locations clearly outside the greater NYC metropolitan area.",
    )

    # 3) Is_Stadium (verify with sources)
    node_stadium = evaluator.add_leaf(
        id="Is_Stadium",
        desc="The identified venue is a stadium (not a non-stadium venue type)",
        parent=task_node,
        critical=True,
    )
    claim_stadium = f"{_safe_name(venue)} is a stadium (an outdoor sports stadium rather than an indoor arena, amphitheater, or other non-stadium venue type)."
    await evaluator.verify(
        claim=claim_stadium,
        node=node_stadium,
        sources=urls,
        additional_instruction="Look for explicit wording that it is a 'stadium' or for context strongly indicating a sports stadium. Do not accept arenas, amphitheaters, or theaters.",
    )

    # 4) Hosts_Concerts (verify with sources)
    node_concerts = evaluator.add_leaf(
        id="Hosts_Concerts",
        desc="The identified stadium hosts concerts",
        parent=task_node,
        critical=True,
    )
    claim_concerts = f"{_safe_name(venue)} hosts concerts (has hosted or is used for large-scale music concerts)."
    await evaluator.verify(
        claim=claim_concerts,
        node=node_concerts,
        sources=urls,
        additional_instruction="Look for mentions of concerts, concert tours, or live music events held at the stadium.",
    )

    # 5) Concert_Seating_Capacity_Provided (existence)
    evaluator.add_custom_node(
        result=_nonempty_str(capacity),
        id="Concert_Seating_Capacity_Provided",
        desc="Answer provides a concert seating capacity figure for the identified stadium",
        parent=task_node,
        critical=True,
    )

    # 6) Reference_URL_Provided (existence)
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="Reference_URL_Provided",
        desc="Answer provides at least one reference URL",
        parent=task_node,
        critical=True,
    )

    # 7) Reference_Supports_Capacity_And_Is_Reliable (split into two critical leaves under a critical parallel node)
    ref_group = evaluator.add_parallel(
        id="Reference_Supports_Capacity_And_Is_Reliable",
        desc="Provided reference sources are reliable and support the stated concert seating capacity figure",
        parent=task_node,
        critical=True,
    )

    # 7a) Capacity supported by reference(s)
    node_capacity_support = evaluator.add_leaf(
        id="Capacity_Supported_By_Reference",
        desc="Provided reference source(s) support the stated concert seating capacity figure",
        parent=ref_group,
        critical=True,
    )
    capacity_str = capacity if _nonempty_str(capacity) else "the stated value"
    claim_capacity = f"The concert seating capacity of {_safe_name(venue)} is {capacity_str}."
    await evaluator.verify(
        claim=claim_capacity,
        node=node_capacity_support,
        sources=urls,
        additional_instruction="Verify that the page(s) explicitly state the concert seating capacity (or an equivalent 'concert configuration' capacity). Allow reasonable textual variants (e.g., 'concert capacity', 'max capacity for concerts', 'capacity including field/floor').",
    )

    # 7b) At least one reference is reliable
    node_ref_reliable = evaluator.add_leaf(
        id="Reference_Is_Reliable",
        desc="At least one provided reference is from a reliable source",
        parent=ref_group,
        critical=True,
    )
    claim_reliable = (
        f"This webpage is a reliable, authoritative source for {_safe_name(venue)}'s concert capacity "
        f"(e.g., the official venue/stadium/operator site, official seating page, or a major reputable publication)."
    )
    await evaluator.verify(
        claim=claim_reliable,
        node=node_ref_reliable,
        sources=urls,
        additional_instruction="Treat official venue/operator/team sites or well-known reputable news/industry sources as reliable. Do not consider random blogs, forums, or user-edited wikis as reliable.",
    )

    # 8) Largest_Capacity_In_Region (verify with sources)
    node_largest = evaluator.add_leaf(
        id="Largest_Capacity_In_Region",
        desc="Among qualifying New York/New Jersey metropolitan-area stadiums that host concerts, the identified stadium has the largest concert seating capacity",
        parent=task_node,
        critical=True,
    )
    claim_largest = (
        f"Among stadiums in the New York/New Jersey metropolitan area that host concerts, {_safe_name(venue)} "
        f"has the largest concert seating capacity."
    )
    await evaluator.verify(
        claim=claim_largest,
        node=node_largest,
        sources=urls,
        additional_instruction="Prefer explicit statements indicating largest/highest concert capacity in the NY/NJ or NYC metro area. Accept clear comparative evidence if provided; if no source substantiates the 'largest' superlative, judge as not supported.",
    )


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
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
    Evaluate an answer for the 'largest concert stadium in NY/NJ metro' task.
    """
    # Initialize evaluator with a parallel root; we will add a critical top-level task node beneath it
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

    # Extract the needed information from the agent's answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueInfo,
        extraction_name="venue_info",
    )

    # Build the verification tree and execute checks
    await build_and_verify(evaluator, root, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()