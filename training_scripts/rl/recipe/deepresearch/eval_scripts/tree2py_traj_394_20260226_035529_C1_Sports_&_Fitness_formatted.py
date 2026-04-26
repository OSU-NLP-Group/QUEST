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
TASK_ID = "mile_high_stadium_info"
TASK_DESCRIPTION = """
What is the name, official seating capacity for football games, and street address of the NFL stadium that is located at exactly 5,280 feet above sea level?
"""

# Optional ground truth for context only (not used to judge)
GROUND_TRUTH_REFERENCE = {
    "known_example": "Empower Field at Mile High (Denver Broncos)",
    "typical_capacity_example": "≈76,125",
    "typical_address_example": "1701 Bryant St, Denver, CO 80204",
    "note": "Ground truth is provided for context only; actual verification relies on the answer's cited sources."
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StadiumExtraction(BaseModel):
    """
    Structured extraction of stadium information from the agent's answer.
    """
    stadium_name: Optional[str] = None
    stadium_name_sources: List[str] = Field(default_factory=list)

    official_seating_capacity: Optional[str] = None
    seating_capacity_sources: List[str] = Field(default_factory=list)

    street_address: Optional[str] = None
    address_sources: List[str] = Field(default_factory=list)

    # Additional sources that explicitly connect the stadium to the 5,280 ft elevation
    elevation_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stadium_info() -> str:
    return """
    Extract the requested details about the NFL stadium that the answer claims is located at exactly 5,280 feet above sea level.
    You must extract ONLY what is explicitly present in the answer text. Do not infer or invent data.

    Required fields:
    1) stadium_name: The name of the stadium (string).
    2) stadium_name_sources: An array of URL strings the answer cites that support the stadium identification (e.g., official site, Wikipedia). Include all URLs explicitly present in the answer that substantively support the stadium identification.
    3) official_seating_capacity: The official seating capacity for football games as stated in the answer (string; keep formatting like commas if present).
    4) seating_capacity_sources: An array of URL strings the answer cites for the seating capacity (e.g., official site page listing capacity, reliable sources).
    5) street_address: The official street address of the stadium as stated in the answer (string; include street number and street name, and if the answer includes city/state/ZIP as part of the street address, keep them).
    6) address_sources: An array of URL strings the answer cites for the street address (e.g., official site contact/location page, reliable sources).
    7) elevation_sources: An array of URL strings explicitly cited by the answer that state the stadium is at exactly 5,280 feet above sea level (allow synonyms like “one mile high (5,280 ft)”).

    URL extraction rules:
    - Extract only URLs explicitly present in the answer. Accept plain URLs and markdown links [text](url).
    - Include full URLs. If a URL is missing protocol, prepend http://.
    - If no sources are mentioned for a field, return an empty array for that field.

    If any required scalar field is missing from the answer, set it to null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_and_dedupe_sources(*lists: List[str]) -> List[str]:
    """Combine multiple URL lists and deduplicate while preserving order."""
    seen = set()
    combined: List[str] = []
    for lst in lists:
        for url in lst or []:
            if not isinstance(url, str):
                continue
            u = url.strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification build                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    extracted: StadiumExtraction,
    root_node,
) -> None:
    """
    Build the verification tree (as per rubric) and perform verifications.
    Root is a critical parallel node with three critical leaf checks:
    - Stadium_Name
    - Seating_Capacity
    - Street_Address
    """
    # Create required critical leaf nodes under the critical parallel root
    stadium_name_node = evaluator.add_leaf(
        id="Stadium_Name",
        desc="Correctly identifies the name of the NFL stadium at 5,280 feet elevation",
        parent=root_node,
        critical=True,
    )

    seating_capacity_node = evaluator.add_leaf(
        id="Seating_Capacity",
        desc="Provides the correct official seating capacity for football games",
        parent=root_node,
        critical=True,
    )

    street_address_node = evaluator.add_leaf(
        id="Street_Address",
        desc="Provides the correct official street address",
        parent=root_node,
        critical=True,
    )

    # Prepare claims and sources
    name_val = extracted.stadium_name or ""
    capacity_val = extracted.official_seating_capacity or ""
    address_val = extracted.street_address or ""

    # For stadium name: verify that the given stadium is explicitly the one at exactly 5,280 ft
    name_sources = _combine_and_dedupe_sources(extracted.stadium_name_sources, extracted.elevation_sources)
    name_claim = (
        f"The NFL stadium located at exactly 5,280 feet above sea level is {name_val}."
    )
    name_instruction = (
        "Use the cited webpages to confirm both the stadium identification and the elevation claim. "
        "Accept reasonable phrasing variations (e.g., 'one mile high (5,280 ft)'). "
        "If none of the cited pages clearly tie the named stadium to 'exactly 5,280 feet', mark as not supported."
    )

    # For seating capacity: confirm the official football seating capacity for the named stadium
    capacity_sources = _combine_and_dedupe_sources(extracted.seating_capacity_sources, extracted.stadium_name_sources)
    capacity_claim = (
        f"The official seating capacity for football games at {name_val} is {capacity_val}."
    )
    capacity_instruction = (
        "Confirm the football seating capacity number on the cited webpages. "
        "Allow minor formatting differences (commas, spaces). "
        "If the pages provide multiple capacities (e.g., concert vs football), choose football. "
        "If the cited pages do not clearly state the capacity, mark as not supported."
    )

    # For street address: confirm the official street address for the named stadium
    address_sources = _combine_and_dedupe_sources(extracted.address_sources, extracted.stadium_name_sources)
    address_claim = (
        f"The official street address of {name_val} is '{address_val}'."
    )
    address_instruction = (
        "Verify the official street address on the cited webpages. "
        "Minor formatting differences (e.g., 'Street' vs 'St', inclusion of city/state/ZIP) are acceptable "
        "as long as the core street address matches. "
        "If the pages do not clearly state the official address, mark as not supported."
    )

    # Run three verifications concurrently
    await evaluator.batch_verify(
        [
            (name_claim, name_sources if name_sources else None, stadium_name_node, name_instruction),
            (capacity_claim, capacity_sources if capacity_sources else None, seating_capacity_node, capacity_instruction),
            (address_claim, address_sources if address_sources else None, street_address_node, address_instruction),
        ]
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
    Evaluate the agent's answer for the NFL stadium at exactly 5,280 ft task.
    Returns a structured summary with a verification tree.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root is parallel per rubric
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

    # Make the root node critical to match rubric (All children must also be critical)
    root.critical = True

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_stadium_info(),
        template_class=StadiumExtraction,
        extraction_name="stadium_extraction",
    )

    # Add ground truth reference info (for context only)
    evaluator.add_ground_truth(
        {
            "reference": GROUND_TRUTH_REFERENCE,
            "task_focus": [
                "Stadium name of NFL venue at 5,280 ft",
                "Official seating capacity for football games",
                "Official street address",
            ],
        },
        gt_type="reference_context",
    )

    # Build tree and verify
    await build_and_verify_tree(evaluator, extracted, root)

    # Return summary
    return evaluator.get_summary()