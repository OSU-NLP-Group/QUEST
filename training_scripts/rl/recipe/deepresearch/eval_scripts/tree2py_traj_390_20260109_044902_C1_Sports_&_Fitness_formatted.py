import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "olympic_pool_depth_min_requirement"
TASK_DESCRIPTION = "What is the minimum pool depth requirement, in meters, for swimming pools used at Olympic Games according to World Aquatics' current guidelines? Provide the depth value and a supporting reference URL."


class PoolDepthExtraction(BaseModel):
    depth_value_meters: Optional[str] = None
    mentions_world_aquatics_or_fina: Optional[bool] = None
    mentions_olympic_games: Optional[bool] = None
    reference_urls: List[str] = Field(default_factory=list)


def prompt_extract_pool_depth_info() -> str:
    return """
    From the answer, extract the following fields related to the minimum pool depth requirement for Olympic Games swimming pools:

    1) depth_value_meters: Extract the stated minimum pool depth number as a string in meters (e.g., "2.5"). 
       - If the answer states the value with units like "2.5 m" or "2.5 meters", return only the numeric part as a string ("2.5").
       - If the value is given as "at least 2.5 m", still return "2.5".
       - If not stated, return null.

    2) mentions_world_aquatics_or_fina: Does the answer explicitly attribute the requirement to "World Aquatics" or "FINA" (former name)? 
       - Return true or false. 
       - Consider "World Aquatics" and "FINA" as equivalent names for the governing body.

    3) mentions_olympic_games: Does the answer clearly specify that the requirement applies to swimming pools used at "Olympic Games" competitions? 
       - Return true or false. 
       - Accept equivalent phrasing like "Olympic competition pools", "Olympic Games pools", etc.

    4) reference_urls: Extract all reference URLs cited in the answer that are intended to support the stated requirement. 
       - Return an array of URLs. 
       - Extract only URLs explicitly present in the answer text (plain URL or markdown link), do not invent any.
       - If no URLs are provided, return an empty array.
    """


async def build_verification_tree(evaluator: Evaluator, extraction: PoolDepthExtraction) -> None:
    complete_node = evaluator.add_parallel(
        id="Complete_Pool_Depth_Information_Provided",
        desc="The answer provides the minimum pool depth requirement (per current World Aquatics guidelines) for Olympic Games swimming pools, including a supporting reference URL.",
        parent=evaluator.root,
        critical=True
    )

    leaf_min_depth = evaluator.add_leaf(
        id="Minimum_Depth_Equals_2_5_Meters",
        desc="States that the minimum required pool depth is 2.5 meters (reported in meters).",
        parent=complete_node,
        critical=True
    )
    claim_min_depth = "The answer explicitly states that the minimum required pool depth is 2.5 meters."
    await evaluator.verify(
        claim=claim_min_depth,
        node=leaf_min_depth,
        additional_instruction="Check the answer text only. Accept minor variants such as '2.5 m', '2.50 m', or phrasing like 'at least 2.5 m' if it clearly indicates the minimum requirement equals 2.5 meters."
    )

    leaf_scope_attr = evaluator.add_leaf(
        id="Applies_To_Olympic_Games_Swimming_Pools_Per_World_Aquatics_Current_Guidelines",
        desc="Clearly attributes the requirement to World Aquatics' current guidelines and specifies that it applies to swimming pools used for Olympic Games competitions.",
        parent=complete_node,
        critical=True
    )
    claim_scope = ("The answer clearly attributes the requirement to World Aquatics (or FINA as the former name) and "
                   "specifies that it applies to swimming pools used for Olympic Games competitions.")
    await evaluator.verify(
        claim=claim_scope,
        node=leaf_scope_attr,
        additional_instruction="Verify using the answer text only. Treat 'World Aquatics' and 'FINA' as equivalent. Accept reasonable synonyms for 'Olympic Games' such as 'Olympic competition pools'."
    )

    leaf_supporting_url = evaluator.add_leaf(
        id="Supporting_Reference_URL_Provided",
        desc="Provides a URL that supports the stated 2.5-meter minimum depth requirement for Olympic Games swimming pools.",
        parent=complete_node,
        critical=True
    )
    claim_support_url = ("At least one of the provided reference URLs explicitly supports that the minimum required pool depth "
                         "is 2.5 meters for swimming pools used at Olympic Games according to World Aquatics (or FINA) guidelines.")
    await evaluator.verify(
        claim=claim_support_url,
        node=leaf_supporting_url,
        sources=extraction.reference_urls,
        additional_instruction=("Use the provided URL(s). If none are provided, conclude not supported. "
                                "Treat World Aquatics and FINA as equivalent naming. The page should clearly indicate both the 2.5 m minimum and "
                                "that the requirement applies to Olympic Games pools or is framed as the governing body's competition standards used for the Olympics.")
    )


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
        default_model=model
    )

    extraction = await evaluator.extract(
        prompt=prompt_extract_pool_depth_info(),
        template_class=PoolDepthExtraction,
        extraction_name="pool_depth_info"
    )

    await build_verification_tree(evaluator, extraction)

    return evaluator.get_summary()