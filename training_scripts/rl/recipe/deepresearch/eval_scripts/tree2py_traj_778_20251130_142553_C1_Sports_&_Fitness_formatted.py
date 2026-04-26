import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "swift_stadium_2024_2025"
TASK_DESCRIPTION = (
    "What is the name, current seating capacity, and complete address of the home stadium "
    "where NFL running back D'Andre Swift's current team plays their home games during the 2024-2025 season?"
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StadiumDetails(BaseModel):
    """Structured extraction of stadium details from the agent's answer."""
    team_name: Optional[str] = None
    stadium_name: Optional[str] = None
    seating_capacity: Optional[str] = None  # Keep as string to allow ranges/approximate text
    address: Optional[str] = None
    sources: List[str] = Field(default_factory=list)  # URLs explicitly cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stadium_details() -> str:
    return (
        "Extract the stadium details provided in the answer. You must extract exactly what is stated in the answer text.\n"
        "Return the following JSON fields:\n"
        "1) team_name: The current NFL team of D'Andre Swift for the 2024–2025 season as named in the answer.\n"
        "2) stadium_name: The full name of the team's home stadium as named in the answer.\n"
        "3) seating_capacity: The current football seating capacity stated in the answer (keep text as-is, allow commas or qualifiers like 'approx.').\n"
        "4) address: The complete street address of the stadium as presented in the answer (street, city, state, and ZIP if provided).\n"
        "5) sources: A list of all URLs the answer explicitly cites to support the stadium identity, capacity, and/or address. Extract actual URLs only "
        "(including ones inside markdown links). If no URLs are provided in the answer, return an empty list.\n"
        "If any field is missing in the answer, return null for that field (or an empty list for sources).\n"
    )


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_stadium_info(
    evaluator: Evaluator,
    parent_node,
    details: StadiumDetails,
) -> None:
    """
    Build the verification tree nodes for stadium information and perform verifications.
    The top-level Stadium_Information node is critical, and all child leaves are critical.
    """
    # Top-level critical node (parallel aggregation across distinct checks)
    stadium_info_node = evaluator.add_parallel(
        id="Stadium_Information",
        desc="Answer provides the correct home stadium for D'Andre Swift's current team (2024–2025) and includes the requested stadium details with verifiable sourcing.",
        parent=parent_node,
        critical=True,
    )

    # Convenience
    team_part = details.team_name if (details.team_name and details.team_name.strip()) else "D'Andre Swift's current NFL team"
    stadium_part = details.stadium_name if (details.stadium_name and details.stadium_name.strip()) else "(stadium not provided)"
    capacity_part = details.seating_capacity if (details.seating_capacity and details.seating_capacity.strip()) else "(capacity not provided)"
    address_part = details.address if (details.address and details.address.strip()) else "(address not provided)"
    sources_list: List[str] = details.sources or []

    # 1) Home_Stadium_Correct (critical leaf)
    home_stadium_node = evaluator.add_leaf(
        id="Home_Stadium_Correct",
        desc="Correctly identifies the home stadium where D'Andre Swift's current team plays home games during the 2024–2025 season (per constraints).",
        parent=stadium_info_node,
        critical=True,
    )
    home_claim = (
        f"The team '{team_part}' plays its home games at '{stadium_part}' during the 2024–2025 NFL season."
        if details.team_name
        else f"The home stadium named '{stadium_part}' is the correct stadium for D'Andre Swift's current NFL team during the 2024–2025 season."
    )
    await evaluator.verify(
        claim=home_claim,
        node=home_stadium_node,
        sources=sources_list if len(sources_list) > 0 else None,
        additional_instruction=(
            "You must judge using the provided URLs if any. If the provided URLs are missing, invalid, or irrelevant, "
            "consider the claim not supported. Focus on whether the page(s) explicitly confirm the team's home stadium "
            "for the 2024–2025 season. Allow reasonable naming variations."
        ),
    )

    # 2) Stadium_Capacity (critical leaf)
    capacity_node = evaluator.add_leaf(
        id="Stadium_Capacity",
        desc="Provides the stadium’s current football seating capacity.",
        parent=stadium_info_node,
        critical=True,
    )
    capacity_claim = f"The current football seating capacity of '{stadium_part}' is '{capacity_part}'."
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_node,
        sources=sources_list if len(sources_list) > 0 else None,
        additional_instruction=(
            "Use the provided URLs to verify the capacity number. If no sources are provided or the URLs do not mention "
            "capacity, mark as not supported. Minor formatting differences (commas, wording like 'approximately') are acceptable "
            "if the number clearly matches."
        ),
    )

    # 3) Stadium_Address (critical leaf)
    address_node = evaluator.add_leaf(
        id="Stadium_Address",
        desc="Provides the complete street address of the stadium (per constraints).",
        parent=stadium_info_node,
        critical=True,
    )
    address_claim = f"The complete street address of '{stadium_part}' is '{address_part}'."
    await evaluator.verify(
        claim=address_claim,
        node=address_node,
        sources=sources_list if len(sources_list) > 0 else None,
        additional_instruction=(
            "Verify the full street address using the provided URLs. If the answer provides a near match with minor formatting differences "
            "(e.g., 'S' vs 'South', punctuation) but clearly the same address, accept it. If no URLs provide the address or no URLs are given, "
            "mark not supported."
        ),
    )

    # 4) Verifiable_Sources (critical leaf)
    sources_node = evaluator.add_leaf(
        id="Verifiable_Sources",
        desc="Provides reliable citations/URLs sufficient to verify the stadium identity, capacity, and address.",
        parent=stadium_info_node,
        critical=True,
    )
    sources_claim = (
        f"The provided sources are reliable and specifically about '{stadium_part}', and at least one source clearly provides evidence "
        f"for the stadium identity, seating capacity, or the complete address."
    )
    await evaluator.verify(
        claim=sources_claim,
        node=sources_node,
        sources=sources_list if len(sources_list) > 0 else None,
        additional_instruction=(
            "Treat official stadium or team pages, the NFL site, or a well-maintained Wikipedia page as reliable. "
            "If the sources list is empty, invalid, or unrelated to the stadium, mark as not supported. "
            "It is acceptable if different sources cover different facts, as long as they can be used to verify the details."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Entry point to evaluate the agent's answer for the stadium information task.
    """
    # Initialize evaluator with a parallel root
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

    # Extract details from the answer
    details = await evaluator.extract(
        prompt=prompt_extract_stadium_details(),
        template_class=StadiumDetails,
        extraction_name="stadium_details",
    )

    # Optionally record some custom info (e.g., number of sources)
    evaluator.add_custom_info(
        info={"extracted_sources_count": len(details.sources or [])},
        info_type="meta",
        info_name="extraction_stats",
    )

    # Build verification nodes and run checks
    await build_and_verify_stadium_info(evaluator, root, details)

    # Return evaluation summary
    return evaluator.get_summary()