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
TASK_ID = "historic_venue_phl_1857"
TASK_DESCRIPTION = (
    "A historic music venue in Philadelphia, Pennsylvania, first opened its doors in 1857 and was "
    "architecturally modeled after La Scala opera house in Milan, Italy. This venue holds the distinction "
    "of being the oldest continuously operating concert hall in the United States that is still used for "
    "its original purpose as a performance venue. What is the name of this historic venue?"
)


# --------------------------------------------------------------------------- #
# Data models for extracting from the agent's answer                          #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt builders                                                  #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    Extract from the answer the single, final venue the answer claims satisfies the task criteria.
    Return a JSON object with:
    - name: The name of the identified historic music venue (string). It should be the main/final venue given by the answer. If multiple candidates are mentioned, choose the one the answer ultimately concludes as the correct venue. If none is provided, return null.
    - sources: An array of all URLs explicitly mentioned in the answer that support or are associated with this identified venue (for example, Wikipedia page, official site, history pages, news, etc.). Extract actual URLs even if embedded in markdown links. Include at most the first 8 URLs. If no URLs are present in the answer, return an empty array.

    Special rules:
    - Only extract URLs explicitly present in the answer text. Do not invent or infer URLs.
    - If a URL is missing a protocol, prepend http://.
    - Deduplicate exact duplicate URLs while preserving order of first occurrence.
    """


# --------------------------------------------------------------------------- #
# Verification sub-tree construction                                          #
# --------------------------------------------------------------------------- #
async def build_historic_venue_verifications(
    evaluator: Evaluator,
    parent_node,
    extracted: VenueExtraction,
) -> None:
    """
    Build the verification nodes according to the rubric tree and dispatch verifications.
    """
    # Parent aggregation node (critical, parallel)
    main_node = evaluator.add_parallel(
        id="Historic_Venue_Identification",
        desc="Correctly identifies the historic music venue matching the specified criteria and provides its name.",
        parent=parent_node,
        critical=True
    )

    # Check that the venue name is provided (custom binary check)
    has_name = bool(extracted.name and extracted.name.strip())
    evaluator.add_custom_node(
        result=has_name,
        id="Venue_Name_Provided",
        desc="The answer provides the name of the venue.",
        parent=main_node,
        critical=True
    )

    venue_name = extracted.name.strip() if extracted.name else ""
    urls = extracted.sources or []

    # Create leaf nodes for each criterion (all critical under the critical parent)
    node_opening = evaluator.add_leaf(
        id="Opening_Year",
        desc="The identified venue opened in 1857.",
        parent=main_node,
        critical=True
    )
    node_location = evaluator.add_leaf(
        id="Location",
        desc="The identified venue is located in Philadelphia, Pennsylvania.",
        parent=main_node,
        critical=True
    )
    node_continuous = evaluator.add_leaf(
        id="Continuous_Original_Use",
        desc="The identified venue is still continuously operating for its original purpose as a performance venue.",
        parent=main_node,
        critical=True
    )
    node_oldest = evaluator.add_leaf(
        id="Oldest_Continuously_Operating_Distinction",
        desc="The identified venue is recognized as the oldest continuously operating concert hall/opera house in the United States still used for its original purpose.",
        parent=main_node,
        critical=True
    )
    node_arch = evaluator.add_leaf(
        id="Architectural_Model",
        desc="The identified venue was architecturally modeled after La Scala opera house in Milan, Italy.",
        parent=main_node,
        critical=True
    )

    claims_and_sources = [
        (
            f"The venue named '{venue_name}' opened in the year 1857.",
            urls,
            node_opening,
            "Verify that the page explicitly states the opening year as 1857 (phrases like 'opened in 1857' or 'built 1855–1857 and opened in 1857' are acceptable)."
        ),
        (
            f"The venue named '{venue_name}' is located in Philadelphia, Pennsylvania.",
            urls,
            node_location,
            "Accept references such as 'Philadelphia', 'Philadelphia, PA', or 'Philadelphia, Pennsylvania'."
        ),
        (
            f"The venue named '{venue_name}' is in continuous operation and is still used for its original purpose as a performance venue (opera/concerts/performances).",
            urls,
            node_continuous,
            "Look for phrases like 'continuously operating', 'still used for its original purpose', 'still hosts performances', or equivalent."
        ),
        (
            f"The venue named '{venue_name}' is the oldest continuously operating concert hall or opera house in the United States that is still used for its original purpose.",
            urls,
            node_oldest,
            "The text must clearly claim it is the 'oldest continuously operating' in the U.S. (not merely 'one of the oldest'). Also ensure it is still used for its original purpose."
        ),
        (
            f"The venue named '{venue_name}' was architecturally modeled after La Scala opera house in Milan, Italy.",
            urls,
            node_arch,
            "Allow synonymous phrasing such as 'patterned after' or 'inspired by La Scala', as long as the modeling intent is explicit."
        ),
    ]

    # Dispatch the five verifications in parallel
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the historic Philadelphia venue identification task.
    """
    # Initialize evaluator with a parallel root (only one main child here)
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

    # Extract venue info (name + any cited URLs)
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction"
    )

    # Optional: record helpful GT/reference info (not used for scoring)
    evaluator.add_ground_truth(
        {
            "hint_expected_venue": "Academy of Music (Philadelphia)",
            "key_facts": [
                "Opened in 1857",
                "Located in Philadelphia, Pennsylvania",
                "Modeled after La Scala (Milan, Italy)",
                "Oldest continuously operating opera house in the U.S. still used for its original purpose"
            ]
        },
        gt_type="reference_info"
    )

    # Build verification tree according to rubric and run checks
    await build_historic_venue_verifications(evaluator, root, extracted_info)

    # Return the structured evaluation summary
    return evaluator.get_summary()