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
TASK_ID = "national_dog_show_2025_info"
TASK_DESCRIPTION = (
    "Provide comprehensive information about the 2025 National Dog Show broadcast on Thanksgiving Day, "
    "including: the broadcast date, the exact broadcast time (in ET), the television network that aired it, "
    "available streaming platforms, the Best in Show winner's name, the winner's breed, the handler's name, "
    "and the location where the event was taped."
)

# Ground truth expectations for verification
EXPECTED_INFO = {
    "broadcast_date": "November 27, 2025",
    "broadcast_time_et": "12:00 p.m. to 2:00 p.m. ET",  # allow equivalent "noon to 2 p.m. ET"
    "tv_network": "NBC",
    "streaming_platforms": ["Peacock", "NBCSports.com", "NBC Sports app"],
    "best_in_show_winner": "Soleil",
    "winner_breed": "Belgian Sheepdog",
    "handler_name": "Daniel Martin",
    "taping_location": "Greater Philadelphia Expo Center"
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DogShowInfo(BaseModel):
    broadcast_date: Optional[str] = None
    broadcast_time_et: Optional[str] = None
    tv_network: Optional[str] = None
    streaming_platforms: List[str] = Field(default_factory=list)
    best_in_show_winner: Optional[str] = None
    winner_breed: Optional[str] = None
    handler_name: Optional[str] = None
    taping_location: Optional[str] = None

    # Per-field sources (URLs explicitly cited in the answer)
    sources_date: List[str] = Field(default_factory=list)
    sources_time: List[str] = Field(default_factory=list)
    sources_network: List[str] = Field(default_factory=list)
    sources_streaming: List[str] = Field(default_factory=list)
    sources_winner: List[str] = Field(default_factory=list)
    sources_breed: List[str] = Field(default_factory=list)
    sources_handler: List[str] = Field(default_factory=list)
    sources_location: List[str] = Field(default_factory=list)

    # General sources section (catch-all URLs cited anywhere in the answer)
    general_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_dog_show_info() -> str:
    return """
    Extract the 2025 National Dog Show Thanksgiving broadcast information exactly as presented in the answer.
    Required fields:
    - broadcast_date: The calendar date stated for the Thanksgiving broadcast (e.g., "November 27, 2025")
    - broadcast_time_et: The exact broadcast time window in Eastern Time (e.g., "12:00 p.m. to 2:00 p.m. ET", "noon to 2 p.m. ET")
    - tv_network: The television network that aired the broadcast (e.g., "NBC")
    - streaming_platforms: A list of platform names mentioned for streaming the broadcast
      Examples of platform names: "Peacock", "NBCSports.com", "NBC Sports app"
      Do not include vague phrases; include clear platform names if stated.
    - best_in_show_winner: The Best in Show winner's call name (e.g., "Soleil")
    - winner_breed: The breed of the Best in Show winner (e.g., "Belgian Sheepdog")
    - handler_name: The handler's full name (e.g., "Daniel Martin")
    - taping_location: The location where the event was taped (e.g., "Greater Philadelphia Expo Center")

    Also extract URLs explicitly cited in the answer for each field:
    - sources_date, sources_time, sources_network, sources_streaming,
      sources_winner, sources_breed, sources_handler, sources_location
    Each of these must be an array of URLs (strings). If the answer provides no URL for that field, return an empty array.

    Finally, extract a 'general_sources' array that lists any URLs cited anywhere in the answer (including a dedicated Sources section).

    Rules:
    - Return null for any missing scalar field. For lists, return an empty array if not present.
    - Extract URLs exactly as shown in the answer. Accept plain URLs and markdown links; include the actual URL.
    - Do not invent any information or URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def pick_sources(primary: List[str], general: List[str]) -> Optional[List[str]]:
    """Prefer field-specific sources; otherwise fall back to general sources; return None if none available."""
    if primary and len(primary) > 0:
        return primary
    if general and len(general) > 0:
        return general
    return None


# --------------------------------------------------------------------------- #
# Main verification builder                                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_nodes(evaluator: Evaluator, root_node, info: DogShowInfo) -> None:
    """
    Build leaf nodes per rubric and execute verifications.
    Note: All children under root are critical; a failure may cause subsequent critical verifications to be skipped.
    """

    # 1) Broadcast Date
    date_node = evaluator.add_leaf(
        id="Broadcast_Date",
        desc="The broadcast date must be November 27, 2025 (Thanksgiving Day)",
        parent=root_node,
        critical=True
    )
    claim_date = f"The broadcast date for the 2025 National Dog Show Thanksgiving broadcast was {EXPECTED_INFO['broadcast_date']}."
    await evaluator.verify(
        claim=claim_date,
        node=date_node,
        sources=pick_sources(info.sources_date, info.general_sources),
        additional_instruction="Accept reasonable date formats or abbreviations such as 'Nov. 27, 2025'. Verify that the answer and/or cited sources explicitly indicate the Thanksgiving broadcast date as November 27, 2025."
    )

    # 2) Broadcast Time (ET)
    time_node = evaluator.add_leaf(
        id="Broadcast_Time",
        desc="The broadcast time must be 12:00 p.m. to 2:00 p.m. ET (or noon to 2 p.m. ET)",
        parent=root_node,
        critical=True
    )
    claim_time = "The broadcast aired from 12:00 p.m. to 2:00 p.m. ET (equivalently, noon to 2:00 p.m. ET)."
    await evaluator.verify(
        claim=claim_time,
        node=time_node,
        sources=pick_sources(info.sources_time, info.general_sources),
        additional_instruction="Treat '12:00 p.m. to 2:00 p.m. ET' and 'noon to 2 p.m. ET' as equivalent. Minor variations in punctuation or spacing are acceptable. Verify that the answer and/or sources explicitly indicate this ET window."
    )

    # 3) Broadcasting Network
    network_node = evaluator.add_leaf(
        id="Broadcasting_Network",
        desc="The television network that aired it must be NBC",
        parent=root_node,
        critical=True
    )
    claim_network = "The television network that aired the 2025 National Dog Show Thanksgiving broadcast was NBC."
    await evaluator.verify(
        claim=claim_network,
        node=network_node,
        sources=pick_sources(info.sources_network, info.general_sources),
        additional_instruction="Allow minor variants like 'NBC-TV' or 'NBC network'. The essence must be that NBC broadcast the show."
    )

    # 4) Streaming Platforms
    streaming_node = evaluator.add_leaf(
        id="Streaming_Platforms",
        desc="The available streaming platforms must include Peacock, NBCSports.com, and the NBC Sports app",
        parent=root_node,
        critical=True
    )
    claim_streaming = (
        "The available streaming platforms for the 2025 National Dog Show Thanksgiving broadcast included "
        "Peacock, NBCSports.com, and the NBC Sports app."
    )
    await evaluator.verify(
        claim=claim_streaming,
        node=streaming_node,
        sources=pick_sources(info.sources_streaming, info.general_sources),
        additional_instruction=(
            "The statement must indicate that all three platforms were available: 'Peacock', 'NBCSports.com', and the 'NBC Sports app'. "
            "Allow minor variants (e.g., 'NBC Sports App' case changes) and URLs pointing to NBCSports.com."
        )
    )

    # 5) Best in Show Winner Name
    winner_name_node = evaluator.add_leaf(
        id="Winner_Name",
        desc="The Best in Show winner's name must be Soleil",
        parent=root_node,
        critical=True
    )
    claim_winner = "The Best in Show winner's name was Soleil."
    await evaluator.verify(
        claim=claim_winner,
        node=winner_name_node,
        sources=pick_sources(info.sources_winner, info.general_sources),
        additional_instruction="Allow case-insensitive matching and minor formatting differences. Verify that the answer and/or sources explicitly identify 'Soleil' as the Best in Show winner."
    )

    # 6) Winner Breed
    breed_node = evaluator.add_leaf(
        id="Winner_Breed",
        desc="The winner's breed must be Belgian Sheepdog",
        parent=root_node,
        critical=True
    )
    claim_breed = "The Best in Show winner's breed was Belgian Sheepdog."
    await evaluator.verify(
        claim=claim_breed,
        node=breed_node,
        sources=pick_sources(info.sources_breed, info.general_sources),
        additional_instruction=(
            "Allow synonymous phrasing like 'Belgian Shepherd Dog (Groenendael)' when clearly referring to Belgian Sheepdog. "
            "The essential breed identification must correspond to Belgian Sheepdog."
        )
    )

    # 7) Handler Name
    handler_node = evaluator.add_leaf(
        id="Handler_Name",
        desc="The handler's name must be Daniel Martin",
        parent=root_node,
        critical=True
    )
    claim_handler = "The handler's name was Daniel Martin."
    await evaluator.verify(
        claim=claim_handler,
        node=handler_node,
        sources=pick_sources(info.sources_handler, info.general_sources),
        additional_instruction="Allow reasonable variants like 'Dan Martin' if the source makes clear it refers to the same person. Verify the handler name explicitly."
    )

    # 8) Taping Location
    location_node = evaluator.add_leaf(
        id="Taping_Location",
        desc="The event taping location must be the Greater Philadelphia Expo Center",
        parent=root_node,
        critical=True
    )
    claim_location = "The event was taped at the Greater Philadelphia Expo Center."
    await evaluator.verify(
        claim=claim_location,
        node=location_node,
        sources=pick_sources(info.sources_location, info.general_sources),
        additional_instruction=(
            "Allow variants indicating 'Greater Philadelphia Expo Center at Oaks' or references to Oaks, PA, "
            "as long as the location clearly corresponds to the Greater Philadelphia Expo Center."
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
    Evaluate an answer for the 2025 National Dog Show Thanksgiving broadcast information task.
    Returns a structured summary with the verification tree and final score.
    """
    # Initialize evaluator with root as parallel aggregation
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

    # Extract information from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_dog_show_info(),
        template_class=DogShowInfo,
        extraction_name="dog_show_info"
    )

    # Record ground truth expectations in summary
    evaluator.add_ground_truth({
        "expected": EXPECTED_INFO,
        "task": "2025 National Dog Show Thanksgiving broadcast — required facts"
    })

    # Build tree and run verifications
    await build_and_verify_nodes(evaluator, root, extracted_info)

    # Return structured result
    return evaluator.get_summary()