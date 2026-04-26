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
TASK_ID = "denver_airport_terminal_hotel"
TASK_DESCRIPTION = (
    "What is the name of the hotel at Denver International Airport that is directly connected "
    "to the airport terminal and does not require shuttle service to access?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HotelExtraction(BaseModel):
    """
    Structured extraction of the named hotel and any URLs cited in the answer.
    """
    hotel_name: Optional[str] = None
    hotel_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel_info() -> str:
    return """
    Extract the hotel's name and any URLs cited in the answer that are intended to support the claim
    that the hotel is directly connected to Denver International Airport's terminal (no shuttle required).

    Return a JSON object with the following fields:
    - hotel_name: The exact name of the hotel provided in the answer. If no hotel is named, return null.
    - hotel_urls: An array of all URLs explicitly mentioned in the answer as sources or references for the hotel.
                  Include URLs in plain form or inside markdown links. If no URLs are provided, return an empty array.

    Rules:
    - Do not invent or infer any hotel names or URLs; extract only what is explicitly present in the answer.
    - For URLs, extract only valid, complete URLs. If a URL is missing a protocol, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Verification sub-tree construction                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    parent_node,
    extraction: HotelExtraction,
) -> None:
    """
    Build and execute the verification checks according to the rubric:
    - Sequential critical main node
      - Critical: Hotel_Name (presence)
      - Critical parallel: Meets_Stated_Constraints
          - Critical leaf: Located_At_DEN
          - Critical leaf: Direct_Terminal_Connection_No_Shuttle_Walkable
    """
    # Create the main sequential, critical node to mirror rubric root
    main_node = evaluator.add_sequential(
        id="denver_airport_terminal_connected_hotel",
        desc="Provide the name of the hotel at Denver International Airport that is directly connected to the terminal (no shuttle required).",
        parent=parent_node,
        critical=True,
    )

    # Leaf 1: Hotel_Name (critical) — we treat it as an existence check
    name_present = bool(extraction.hotel_name and extraction.hotel_name.strip())
    evaluator.add_custom_node(
        result=name_present,
        id="hotel_name",
        desc="State the hotel’s name.",
        parent=main_node,
        critical=True,
    )

    # Node 2: Meets_Stated_Constraints (critical, parallel)
    constraints_node = evaluator.add_parallel(
        id="meets_stated_constraints",
        desc="Verify the named hotel satisfies the question constraints.",
        parent=main_node,
        critical=True,
    )

    # Prepare sources (can be None or list)
    sources: List[str] = extraction.hotel_urls if extraction.hotel_urls else []
    hotel_name = extraction.hotel_name or ""

    # Leaf 2a: Located_At_DEN (critical)
    located_node = evaluator.add_leaf(
        id="located_at_den",
        desc="The named hotel is located at Denver International Airport (DEN).",
        parent=constraints_node,
        critical=True,
    )
    located_claim = (
        f"The hotel '{hotel_name}' is located at Denver International Airport (DEN), "
        f"on airport property or directly at the terminal complex."
    )
    await evaluator.verify(
        claim=located_claim,
        node=located_node,
        sources=sources if sources else None,
        additional_instruction=(
            "Use the provided URLs (if any) and the answer context to determine whether the hotel is located at Denver "
            "International Airport. Accept explicit phrases like 'Westin Denver International Airport', 'on airport property', "
            "'at DEN', or 'adjacent to the terminal complex'. If the URLs are irrelevant or do not support the claim, mark as not supported."
        ),
    )

    # Leaf 2b: Direct_Terminal_Connection_No_Shuttle_Walkable (critical)
    connection_node = evaluator.add_leaf(
        id="terminal_connection_walkable_no_shuttle",
        desc="The named hotel is directly connected to the airport terminal and is accessible on foot from the terminal without requiring shuttle service or external transportation.",
        parent=constraints_node,
        critical=True,
    )
    connection_claim = (
        f"The hotel '{hotel_name}' is physically connected to the terminal at Denver International Airport and can be "
        f"accessed on foot from the terminal without requiring shuttle service."
    )
    await evaluator.verify(
        claim=connection_claim,
        node=connection_node,
        sources=sources if sources else None,
        additional_instruction=(
            "Look for clear indications in the provided URLs (if any) that the hotel is connected to the terminal via walkway/bridge "
            "or is physically attached. Accept phrases such as 'connected to the terminal', 'walkable from the terminal', 'no shuttle required', "
            "or 'steps from the terminal'. If available sources contradict or fail to support this, mark as not supported."
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
    Evaluate the answer for the Denver airport terminal-connected hotel task.
    """
    # Initialize evaluator; use sequential root for clarity (root remains non-critical internally)
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
        default_model=model,
    )

    # Extract hotel name and cited URLs from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_hotel_info(),
        template_class=HotelExtraction,
        extraction_name="hotel_extraction",
    )

    # Optional: record custom info for debugging/summary
    evaluator.add_custom_info(
        info={
            "extracted_hotel_name": extraction.hotel_name,
            "extracted_urls_count": len(extraction.hotel_urls),
            "extracted_urls": extraction.hotel_urls,
        },
        info_type="extraction_debug",
        info_name="hotel_extraction_debug",
    )

    # Build and run verification tree
    await build_verification_tree(evaluator, root, extraction)

    # Return evaluation summary
    return evaluator.get_summary()