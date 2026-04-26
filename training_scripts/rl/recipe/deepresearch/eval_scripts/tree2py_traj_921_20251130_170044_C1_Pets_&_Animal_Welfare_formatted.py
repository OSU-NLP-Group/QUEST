import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "2025_national_dog_show_handler"
TASK_DESCRIPTION = "Who was the handler of the Best in Show winner at the 2025 National Dog Show, and what city and state is the handler from?"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HandlerInfo(BaseModel):
    handler_name: Optional[str] = None
    handler_city: Optional[str] = None
    handler_state: Optional[str] = None
    sources: List[str] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_handler_info() -> str:
    return (
        "From the provided answer, extract the following fields about the handler of the Best in Show winner at the 2025 National Dog Show:\n"
        "1) handler_name: The full name of the handler who presented the Best in Show winner.\n"
        "2) handler_city: The city the handler is from (hometown or residence).\n"
        "3) handler_state: The state the handler is from.\n"
        "4) sources: A list of URLs explicitly cited in the answer that support any of these facts, especially the handler's name and hometown.\n"
        "Rules:\n"
        "- Only extract information explicitly present in the answer text.\n"
        "- If a field is missing, set it to null (or an empty list for sources).\n"
        "- For URLs, extract actual URLs provided (including markdown links); do not invent any.\n"
        "- If the city and state appear together (e.g., \"City, State\"), split them accordingly.\n"
    )

# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_handler_info(
    evaluator: Evaluator,
    parent_node,
    extracted: HandlerInfo,
) -> None:
    """
    Build critical parallel verification nodes:
    - Handler_Name
    - Handler_City
    - Handler_State
    Each leaf is verified against the cited sources when available.
    """
    # Create the critical parent node under root (since root itself is non-critical by design)
    info_node = evaluator.add_parallel(
        id="2025_National_Dog_Show_Handler_Information",
        desc="Verify the handler name and hometown details (city and state) for the Best in Show winner at the 2025 National Dog Show.",
        parent=parent_node,
        critical=True,
    )

    sources = extracted.sources if extracted and extracted.sources else None
    name_val = (extracted.handler_name or "").strip()
    city_val = (extracted.handler_city or "").strip()
    state_val = (extracted.handler_state or "").strip()

    # 1) Handler Name leaf
    name_node = evaluator.add_leaf(
        id="Handler_Name",
        desc="Correctly identifies the full name of the handler who presented the Best in Show winner at the 2025 National Dog Show.",
        parent=info_node,
        critical=True,
    )
    name_claim = (
        f"The handler who presented the Best in Show winner at the 2025 National Dog Show is {name_val}."
        if name_val
        else "The handler who presented the Best in Show winner at the 2025 National Dog Show is explicitly identified."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_node,
        sources=sources,
        additional_instruction=(
            "Verify that the cited webpage(s) explicitly mention the 2025 National Dog Show Best in Show winner "
            "and name the handler. Do not confuse this with other shows (e.g., Westminster). Minor formatting variations "
            "in the handler's name (middle initials, casing) are acceptable as long as the person is clearly the same."
        ),
    )

    # 2) Handler City leaf
    city_node = evaluator.add_leaf(
        id="Handler_City",
        desc="Correctly specifies the city the handler is from.",
        parent=info_node,
        critical=True,
    )
    city_claim = (
        f"The handler {name_val} is from the city of {city_val}."
        if city_val and name_val
        else (
            f"The handler is from the city of {city_val}."
            if city_val
            else "The handler's city of origin is explicitly provided."
        )
    )
    await evaluator.verify(
        claim=city_claim,
        node=city_node,
        sources=sources,
        additional_instruction=(
            "Confirm that the cited webpage(s) indicate the handler's hometown/residence city. It may appear as "
            "\"City, State\" on the page; consider the city correctly specified if it matches the city portion in such a format."
        ),
    )

    # 3) Handler State leaf
    state_node = evaluator.add_leaf(
        id="Handler_State",
        desc="Correctly specifies the state the handler is from.",
        parent=info_node,
        critical=True,
    )
    state_claim = (
        f"The handler {name_val} is from the state of {state_val}."
        if state_val and name_val
        else (
            f"The handler is from the state of {state_val}."
            if state_val
            else "The handler's state of origin is explicitly provided."
        )
    )
    await evaluator.verify(
        claim=state_claim,
        node=state_node,
        sources=sources,
        additional_instruction=(
            "Confirm that the cited webpage(s) indicate the handler's state. Allow common abbreviations "
            "(e.g., 'PA' for 'Pennsylvania') or minor variations if they clearly refer to the same state."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate the answer for the 2025 National Dog Show handler information task.
    """
    # Initialize evaluator
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

    # Extract handler info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_handler_info(),
        template_class=HandlerInfo,
        extraction_name="handler_info",
    )

    # Build verification nodes and run checks
    await verify_handler_info(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()