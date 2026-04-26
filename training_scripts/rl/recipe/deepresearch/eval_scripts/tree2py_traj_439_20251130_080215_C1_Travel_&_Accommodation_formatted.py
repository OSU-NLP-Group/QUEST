import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "disney_destiny_departure_maiden_2025"
TASK_DESCRIPTION = "From which port facility and city will the Disney Destiny cruise ship depart for its maiden voyage in November 2025?"


class DepartureExtraction(BaseModel):
    port_facility: Optional[str] = None
    city: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


def prompt_extract_departure() -> str:
    return """
    Extract from the answer the specific departure information for Disney Destiny’s maiden voyage scheduled for November 2025.
    Return the following fields:
    1) port_facility: The name of the port facility (e.g., "Port Everglades"), exactly as written in the answer.
    2) city: The name of the departure city (e.g., "Fort Lauderdale"), exactly as written in the answer.
    3) source_urls: All URLs explicitly cited in the answer that are presented as sources for the departure details. Extract actual URLs only (including those inside markdown links). If no URLs are cited, return an empty list.
    If either port_facility or city is not mentioned, set it to null.
    """


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

    extracted = await evaluator.extract(
        prompt=prompt_extract_departure(),
        template_class=DepartureExtraction,
        extraction_name="departure_extraction"
    )

    evaluator.add_ground_truth({
        "expected_port_facility": "Port Everglades",
        "expected_city": "Fort Lauderdale",
        "event": "Disney Destiny maiden voyage (Nov 2025)"
    })

    main_node = evaluator.add_parallel(
        id="Disney_Destiny_Maiden_Voyage_Nov_2025_Departure",
        desc="Evaluates whether the response identifies the correct departure port facility and departure city for Disney Destiny’s maiden voyage in November 2025.",
        parent=root,
        critical=True
    )

    port_leaf = evaluator.add_leaf(
        id="Departure_Port_Facility_Is_Port_Everglades",
        desc="Response identifies the departure port facility as Port Everglades.",
        parent=main_node,
        critical=True
    )
    port_claim = "The answer explicitly identifies the departure port facility for Disney Destiny’s maiden voyage in November 2025 as 'Port Everglades'."
    await evaluator.verify(
        claim=port_claim,
        node=port_leaf,
        sources=None,
        additional_instruction="Focus on the answer content only. Determine whether it clearly states Port Everglades as the departure port facility for Disney Destiny’s maiden voyage (Nov 2025). Allow minor phrasing variations such as 'from Port Everglades', 'at Port Everglades', or 'Port Everglades cruise port'. Do not rely on external knowledge."
    )

    city_leaf = evaluator.add_leaf(
        id="Departure_City_Is_Fort_Lauderdale",
        desc="Response identifies the departure city as Fort Lauderdale.",
        parent=main_node,
        critical=True
    )
    city_claim = "The answer explicitly identifies the departure city for Disney Destiny’s maiden voyage in November 2025 as 'Fort Lauderdale'."
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=None,
        additional_instruction="Focus on the answer content only. Determine whether it clearly states Fort Lauderdale as the departure city for Disney Destiny’s maiden voyage (Nov 2025). Allow minor variations like 'Ft. Lauderdale', 'Fort Lauderdale, FL', or 'Fort Lauderdale, Florida'. Do not rely on external knowledge."
    )

    return evaluator.get_summary()