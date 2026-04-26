import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "midtown_arena_concert_capacity"
TASK_DESCRIPTION = "What is the concert capacity of the major arena in Midtown Manhattan, New York City, that is located directly above Pennsylvania Station?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    """
    Structured extraction of the venue and capacity details as stated in the answer.
    """
    venue_name: Optional[str] = None
    # All URLs in the answer that are used to identify the venue (e.g., Wikipedia, official site).
    venue_urls: List[str] = Field(default_factory=list)
    # All URLs in the answer that are used to support the concert capacity figure.
    capacity_urls: List[str] = Field(default_factory=list)
    # Optional: the capacity figure for concerts as stated in the answer; kept as free text.
    concert_capacity: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    Extract from the answer the following fields related to the question about the major arena in Midtown Manhattan, NYC, located directly above Pennsylvania Station:
    - venue_name: The name of the venue identified by the answer. This should be the main arena addressed by the answer (e.g., Madison Square Garden). Return null if not explicitly named.
    - venue_urls: All URLs cited in the answer that are used to identify or describe the venue (e.g., official site, Wikipedia, venue profile pages). Return an empty list if none.
    - capacity_urls: All URLs cited in the answer that specifically support the concert capacity figure for the venue. If the answer does not separate sources for capacity versus general venue info, include any relevant URLs. Return an empty list if none.
    - concert_capacity: The concert capacity value for the venue as explicitly stated in the answer (for concerts only). Keep it exactly as written (e.g., '20,000', 'about 20,000', 'up to 20,789', etc.). Return null if not provided.

    Notes:
    - Do not invent URLs. Only extract URLs explicitly present in the answer (including markdown links).
    - The concert capacity should be the number for concerts (not basketball, hockey, or other configurations).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _merge_sources(primary: List[str], secondary: List[str]) -> List[str]:
    """
    Merge and de-duplicate two lists of URLs, preserving order (primary first).
    """
    seen = set()
    merged: List[str] = []
    for url in primary + secondary:
        if url and url not in seen:
            merged.append(url)
            seen.add(url)
    return merged


async def verify_concert_arena_information(
    evaluator: Evaluator,
    parent_node,
    extraction: VenueExtraction
) -> None:
    """
    Build and verify the rubric tree for:
    - Venue identification (critical, sequential first)
    - Concert capacity value correctness (critical, sequential second)
    """

    # Create main sequential node (Critical as per rubric)
    main = evaluator.add_sequential(
        id="concert_arena_information",
        desc="Determines the concert capacity of the major arena in Midtown Manhattan, NYC, located directly above Pennsylvania Station.",
        parent=parent_node,
        critical=True
    )

    # Prepare sources for each verification
    sources_for_venue = extraction.venue_urls or extraction.capacity_urls or []
    sources_for_capacity = _merge_sources(extraction.capacity_urls, extraction.venue_urls)

    # ---------------------------
    # 1) Venue Identification (Leaf, Critical)
    # ---------------------------
    venue_ident_node = evaluator.add_leaf(
        id="venue_identification",
        desc="Identifies a venue that is a major concert arena (not a theater/stadium), located in Midtown Manhattan, New York City, and situated directly above Pennsylvania Station.",
        parent=main,
        critical=True
    )

    venue_name_label = extraction.venue_name.strip() if extraction.venue_name else "the identified venue"

    venue_claim = (
        f"{venue_name_label} is a major concert arena in Midtown Manhattan, New York City, and it is situated directly above Pennsylvania Station (Penn Station)."
    )

    await evaluator.verify(
        claim=venue_claim,
        node=venue_ident_node,
        sources=sources_for_venue if sources_for_venue else None,
        additional_instruction=(
            "Verify that the webpage(s) clearly indicate the venue is: "
            "1) an indoor arena (not a theater or open-air stadium), "
            "2) located in Midtown Manhattan in New York City, and "
            "3) physically situated directly above Pennsylvania Station (also known as Penn Station). "
            "Allow reasonable paraphrases such as 'atop', 'above', 'over', or 'built on top of' Penn Station."
        )
    )

    # ---------------------------
    # 2) Concert Capacity Value (Leaf, Critical)
    # ---------------------------
    capacity_node = evaluator.add_leaf(
        id="concert_capacity_value",
        desc="States the official concert capacity figure for concerts for the identified venue.",
        parent=main,
        critical=True
    )

    # Prefer referencing the answer-stated capacity, but allow verifier to read it from the answer context.
    # This avoids over-constraining the exact formatting and leverages the answer as context.
    if extraction.concert_capacity and extraction.concert_capacity.strip():
        capacity_text = extraction.concert_capacity.strip()
        capacity_claim = (
            f"The concert capacity of {venue_name_label} as stated in the answer ('{capacity_text}') is correct according to the cited sources, "
            f"and specifically refers to the capacity for concerts (not basketball, hockey, or other configurations)."
        )
    else:
        # If the answer did not explicitly state a capacity figure, still verify based on the answer text and sources;
        # This will typically fail if no figure is present or cannot be corroborated by sources.
        capacity_claim = (
            f"The answer's stated concert capacity for {venue_name_label} is accurate according to the cited sources, "
            f"and specifically refers to the capacity for concerts (not basketball, hockey, or other configurations)."
        )

    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_node,
        sources=sources_for_capacity if sources_for_capacity else None,
        additional_instruction=(
            "Read the answer text to identify the concert capacity figure it claims. Then check the provided webpage(s) to confirm that the same figure "
            "is given for concerts specifically. If the page lists multiple capacities for different configurations, focus on the 'concerts' capacity. "
            "Allow minor formatting differences (e.g., commas, approximate wording like 'about' or 'up to') and small rounding differences, "
            "but the meaning should clearly match the concerts capacity."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Midtown Manhattan arena concert capacity task.
    """
    # Initialize evaluator with a simple root (root itself remains non-critical)
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
        extraction_name="venue_extraction"
    )

    # Optional: record expected reference info (for debugging in summary)
    evaluator.add_ground_truth({
        "expected_venue_example": "Madison Square Garden",
        "key_requirements": [
            "Indoor arena (not a theater/stadium)",
            "Located in Midtown Manhattan, NYC",
            "Situated directly above Pennsylvania Station",
            "Concert capacity figure must be for concerts"
        ]
    })

    # Build and verify rubric nodes
    await verify_concert_arena_information(evaluator, root, extraction)

    # Return full structured summary with verification tree
    return evaluator.get_summary()