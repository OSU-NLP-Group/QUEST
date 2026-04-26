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
TASK_ID = "dmb_nc_capacity_2026_06_06"
TASK_DESCRIPTION = """
What is the total capacity of the venue where Dave Matthews Band is performing in North Carolina on June 6, 2026?
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueCapacityExtraction(BaseModel):
    """
    Structured extraction from the agent's answer.

    venue_name: The name of the North Carolina venue where DMB performs on June 6, 2026.
    capacity: The total capacity value for the venue as provided in the answer (keep as string to allow '18,500', '18500', '19,000 seats', etc.).
    venue_sources: URLs cited in the answer that support the venue identification for the given date/location.
    capacity_sources: URLs cited in the answer that support the capacity number of the identified venue.
    """
    venue_name: Optional[str] = None
    capacity: Optional[str] = None
    venue_sources: List[str] = Field(default_factory=list)
    capacity_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_capacity() -> str:
    return """
    Extract the following fields strictly from the answer text:

    - venue_name: The name of the venue in North Carolina where Dave Matthews Band is performing on June 6, 2026, as stated in the answer. If not mentioned, return null.
    - capacity: The total capacity of that venue as given in the answer. Keep it as a string exactly as presented (e.g., "18,500", "18500", "19,000 seats"). If missing, return null.
    - venue_sources: A list of all URLs in the answer that directly support which venue the band is playing at on June 6, 2026 in North Carolina (e.g., tour schedule pages, event listings, venue event page). If none provided, return an empty list.
    - capacity_sources: A list of all URLs in the answer that directly support the capacity value for the identified venue (e.g., official venue page, reliable info page, Wikipedia/venue profile). If none provided, return an empty list.

    Notes:
    - Only extract URLs that are explicitly present in the answer (plain URLs or within markdown links).
    - If the answer does not distinguish sources per field, include any general or shared URLs in both venue_sources and capacity_sources if they plausibly support those claims.
    - Do not invent or infer values. If a field is missing, set it to null (or empty list for URLs).
    """.strip()


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _merge_and_dedupe_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst:
            if not isinstance(u, str):
                continue
            u = u.strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extraction: VenueCapacityExtraction) -> None:
    """
    Build the verification tree according to the rubric and run the verifications.
    Tree structure (reflecting the JSON rubric):
    - root (created by evaluator.initialize)
      - Answer_Correctness (sequential, critical)
        - Identify_Correct_Venue (leaf, critical)
        - Provide_Total_Capacity (leaf, critical)
    """
    # Create the main rubric node as a sequential, critical node
    correctness_node = evaluator.add_sequential(
        id="Answer_Correctness",
        desc="Evaluates whether the response answers: the total capacity of the venue where Dave Matthews Band is performing in North Carolina on June 6, 2026.",
        parent=evaluator.root,
        critical=True,
    )

    # 1) Identify_Correct_Venue (Critical leaf)
    identify_node = evaluator.add_leaf(
        id="Identify_Correct_Venue",
        desc="Correctly identifies the specific venue in North Carolina where Dave Matthews Band is performing on June 6, 2026.",
        parent=correctness_node,
        critical=True,
    )

    # Build claim and collect sources for venue identification
    venue_name = extraction.venue_name or ""
    venue_sources = _merge_and_dedupe_urls(extraction.venue_sources, extraction.capacity_sources)

    claim_identify = (
        f"On June 6, 2026, Dave Matthews Band is scheduled to perform in North Carolina at {venue_name}."
    )

    await evaluator.verify(
        claim=claim_identify,
        node=identify_node,
        sources=venue_sources,
        additional_instruction=(
            "Verify that at least one provided source explicitly lists a Dave Matthews Band concert on June 6, 2026 "
            "in North Carolina and names the venue. Allow minor naming variations or aliases for the venue "
            "(e.g., corporate sponsorship names vs. common names), but ensure it is the same physical venue in NC. "
            "If the sources fail to show the specific date, state, and venue, mark as not supported."
        ),
    )

    # 2) Provide_Total_Capacity (Critical leaf)
    capacity_node = evaluator.add_leaf(
        id="Provide_Total_Capacity",
        desc="Provides the venue's total capacity as a numeric value (people/seats), consistent with authoritative venue/concert information.",
        parent=correctness_node,
        critical=True,
    )

    # Build claim and collect sources for capacity
    capacity_value = extraction.capacity or ""
    capacity_sources = _merge_and_dedupe_urls(extraction.capacity_sources, extraction.venue_sources)

    claim_capacity = (
        f"The total capacity of {venue_name} is {capacity_value}."
    )

    await evaluator.verify(
        claim=claim_capacity,
        node=capacity_node,
        sources=capacity_sources,
        additional_instruction=(
            "Check the page(s) for the venue's stated total capacity for concerts. The value in the claim should be a number "
            "(commas or the word 'seats' are acceptable). If multiple configurations are shown, use the standard or maximum "
            "concert capacity reported. Minor rounding differences are acceptable (e.g., 18,500 vs. 18,000) if clearly the same figure. "
            "If the sources do not explicitly support a total capacity close to the claimed number, mark as not supported."
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
    Entry point for evaluating an answer to the task:
    'What is the total capacity of the venue where Dave Matthews Band is performing in North Carolina on June 6, 2026?'
    """
    # Initialize evaluator (root node created internally)
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root can be parallel; rubric node below is sequential
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_venue_capacity(),
        template_class=VenueCapacityExtraction,
        extraction_name="venue_capacity_extraction",
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extraction)

    # Optional: record a concise snapshot of extracted key fields
    evaluator.add_custom_info(
        info={
            "extracted_venue_name": extraction.venue_name,
            "extracted_capacity": extraction.capacity,
            "venue_sources_count": len(extraction.venue_sources),
            "capacity_sources_count": len(extraction.capacity_sources),
        },
        info_type="extraction_summary",
    )

    # Return summary
    return evaluator.get_summary()