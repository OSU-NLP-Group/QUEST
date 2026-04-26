import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "largest_concert_hall_us"
TASK_DESCRIPTION = """
What is the largest concert hall venue in the United States by seating capacity, excluding stadiums and arenas? Provide the venue name, its location (city and state), and the official seating capacity. Include a reference URL to support your answer.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueInfo(BaseModel):
    """Structured info extracted from the agent's answer."""
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    official_capacity: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    Extract the following fields exactly as presented in the answer:
    1. venue_name: The name of the venue the answer claims is the largest concert hall in the U.S. by seating capacity.
    2. city: The city of the venue (if provided).
    3. state: The state of the venue (if provided).
    4. official_capacity: The official seating capacity value that the answer claims for the venue. Extract as text exactly (e.g., "3,004", "2,900+", etc.).
    5. reference_urls: All URLs the answer provides as references. Include every URL explicitly present in the answer (plain URLs or markdown links). If none are provided, return an empty list.

    Rules:
    - Do not invent any information. Only extract what appears in the answer.
    - If a field is not present, set it to null (or empty list for reference_urls).
    - For reference_urls, include only valid-looking URLs (prepend http:// if missing protocol).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: VenueInfo) -> None:
    """
    Construct the verification tree following the rubric and run verifications.
    """
    # Top-level task node (critical, parallel aggregation)
    task_node = evaluator.add_parallel(
        id="Largest_Concert_Hall_Task",
        desc="Identify the largest concert hall venue in the United States by seating capacity (excluding stadiums and arenas) and provide required supporting details (name, city/state, official capacity, and a supporting URL).",
        parent=evaluator.root,
        critical=True
    )

    # ----------------------------- Venue Identification ----------------------------- #
    venue_ident_node = evaluator.add_parallel(
        id="Venue_Identification",
        desc="Correctly identifies the qualifying venue per category and ranking constraints.",
        parent=task_node,
        critical=True
    )

    # Concert hall classification (critical leaf)
    classification_leaf = evaluator.add_leaf(
        id="Concert_Hall_Classification",
        desc="The venue is a concert hall (and not a stadium or arena).",
        parent=venue_ident_node,
        critical=True
    )
    classification_claim = (
        f"The venue '{extracted.venue_name or ''}' is a concert hall (e.g., concert hall, auditorium, opera house, symphony hall, music hall) "
        f"and is not a stadium or arena."
    )
    await evaluator.verify(
        claim=classification_claim,
        node=classification_leaf,
        sources=extracted.reference_urls if extracted.reference_urls else None,
        additional_instruction=(
            "Judge based on the referenced page(s). Accept synonyms like 'concert hall', 'auditorium', 'opera house', "
            "'symphony hall', 'music hall', or a performance hall in a performing arts center. "
            "If the page describes the venue primarily as an arena or stadium used for sports, mark as incorrect."
        ),
    )

    # Largest by seating capacity in US within concert hall category (critical leaf)
    largest_leaf = evaluator.add_leaf(
        id="Largest_By_Seating_Capacity_US",
        desc="The venue is the largest concert hall in the United States by seating capacity (within the concert hall category).",
        parent=venue_ident_node,
        critical=True
    )
    largest_claim = (
        f"Among concert halls in the United States (excluding stadiums and arenas), '{extracted.venue_name or ''}' has the highest seating capacity."
    )
    await evaluator.verify(
        claim=largest_claim,
        node=largest_leaf,
        sources=extracted.reference_urls if extracted.reference_urls else None,
        additional_instruction=(
            "Only mark correct if a credible reference explicitly supports the ranking or a list shows the venue as #1 by seating capacity "
            "within the concert hall category in the United States. General mentions of capacity without comparative ranking are insufficient."
        ),
    )

    # ----------------------------- Location Details ----------------------------- #
    location_node = evaluator.add_parallel(
        id="Location_Details",
        desc="Provides the venue's correct location (city and state).",
        parent=task_node,
        critical=True
    )

    # City (critical leaf)
    city_leaf = evaluator.add_leaf(
        id="City",
        desc="Correct city where the identified venue is located.",
        parent=location_node,
        critical=True
    )
    city_claim = f"The venue '{extracted.venue_name or ''}' is located in the city of {extracted.city or ''}."
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=extracted.reference_urls if extracted.reference_urls else None,
        additional_instruction=(
            "Verify that the reference page mentions the venue and the specified city. "
            "Allow minor variants like borough/neighborhood names if the city is clearly indicated."
        ),
    )

    # State (critical leaf)
    state_leaf = evaluator.add_leaf(
        id="State",
        desc="Correct state where the identified venue is located.",
        parent=location_node,
        critical=True
    )
    state_claim = f"The venue '{extracted.venue_name or ''}' is located in the state of {extracted.state or ''}."
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=extracted.reference_urls if extracted.reference_urls else None,
        additional_instruction=(
            "Verify that the reference page mentions the venue and the specified state (e.g., CA, California). "
            "Allow common abbreviations and minor formatting variants."
        ),
    )

    # ----------------------------- Official Seating Capacity ----------------------------- #
    capacity_leaf = evaluator.add_leaf(
        id="Official_Seating_Capacity",
        desc="Provides the venue's official seating capacity accurately.",
        parent=task_node,
        critical=True
    )
    capacity_claim = f"The official seating capacity of '{extracted.venue_name or ''}' is {extracted.official_capacity or ''}."
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=extracted.reference_urls if extracted.reference_urls else None,
        additional_instruction=(
            "Confirm that the page lists the venue's official seating capacity matching the provided value. "
            "Allow reasonable rounding, but it must clearly correspond to seating capacity (not standing capacity or different configurations)."
        ),
    )

    # ----------------------------- Reference URL ----------------------------- #
    reference_leaf = evaluator.add_leaf(
        id="Reference_URL",
        desc="Provides a valid, accessible reference URL that supports the venue identification and the key factual claims (at minimum capacity, and ideally classification/ranking).",
        parent=task_node,
        critical=True
    )
    # This leaf ensures at least one provided URL is valid and relevant.
    reference_claim = (
        f"The answer provides at least one valid and accessible reference URL that is about '{extracted.venue_name or ''}' "
        f"and corroborates the seating capacity (and ideally the concert hall classification or largest-by-capacity claim)."
    )
    await evaluator.verify(
        claim=reference_claim,
        node=reference_leaf,
        sources=extracted.reference_urls if extracted.reference_urls else None,
        additional_instruction=(
            "If no URL is provided in the answer, mark this claim incorrect. "
            "A valid URL should load and contain content relevant to the venue; ideally it also supports capacity and classification/ranking."
        ),
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
    Evaluate an answer for the largest concert hall task.
    """
    # Initialize evaluator (root uses parallel to match rubric)
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured venue info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueInfo,
        extraction_name="venue_info",
    )

    # Record extracted information for transparency
    evaluator.add_custom_info(
        info={
            "venue_name": extracted_info.venue_name,
            "city": extracted_info.city,
            "state": extracted_info.state,
            "official_capacity": extracted_info.official_capacity,
            "reference_urls": extracted_info.reference_urls,
        },
        info_type="extraction_summary",
        info_name="extracted_venue_info"
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extracted_info)

    # Return standardized summary
    return evaluator.get_summary()