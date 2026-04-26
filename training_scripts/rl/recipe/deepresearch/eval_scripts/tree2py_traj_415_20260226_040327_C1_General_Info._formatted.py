import asyncio
import logging
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "jake_paul_dec_2025_venue_city"
TASK_DESCRIPTION = (
    "In December 2025, Jake Paul participated in a boxing match. What was the name of the venue where this fight took place, "
    "and in which city is this venue located?"
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueCityExtraction(BaseModel):
    venue_name: Optional[str] = None
    city: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_city() -> str:
    return """
    Extract the information about Jake Paul's boxing match that took place in December 2025, as stated in the answer.
    Return the following fields:
    - venue_name: The name of the venue (e.g., "Madison Square Garden", "T-Mobile Arena", etc.) where the December 2025 fight took place.
    - city: The city where this venue is located (e.g., "New York", "Las Vegas"). If the answer provides "City, State" or "City, State, Country", return the city portion (the first city name).
    - source_urls: A list of all URLs explicitly included in the answer that support or are associated with the December 2025 fight info. Only include actual URLs present in the answer text. If none are present, return an empty list.
    
    Important rules:
    - If multiple fights or dates are mentioned, focus ONLY on the fight that took place in December 2025.
    - Do not invent or infer any information not explicitly present in the answer.
    - If a requested field is missing from the answer, set it to null (for strings) or [] (for lists).
    """


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate whether the answer provides both the venue name and the city location for Jake Paul's December 2025 boxing match.
    According to the rubric:
    - Two critical checks (parallel aggregation):
      1) A venue name is provided.
      2) A city location is provided.
    """
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

    # Extract the venue and city from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_venue_city(),
        template_class=VenueCityExtraction,
        extraction_name="venue_city_extraction",
    )

    # Leaf 1: Venue name provided (critical)
    evaluator.add_custom_node(
        result=bool(extraction.venue_name and extraction.venue_name.strip()),
        id="venue_identification",
        desc="A venue name is provided for where Jake Paul's December 2025 fight took place",
        parent=root,
        critical=True,
    )

    # Leaf 2: City provided (critical)
    evaluator.add_custom_node(
        result=bool(extraction.city and extraction.city.strip()),
        id="city_location",
        desc="A city location is provided for the venue",
        parent=root,
        critical=True,
    )

    return evaluator.get_summary()