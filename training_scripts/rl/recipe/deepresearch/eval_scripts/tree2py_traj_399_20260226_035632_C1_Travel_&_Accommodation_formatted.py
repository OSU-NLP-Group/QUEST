import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "mia_phase2_restrooms"
TASK_DESCRIPTION = "Miami International Airport is undergoing a comprehensive restroom modernization project across its terminals and concourses. What is the scheduled completion date for Phase 2 of this restroom modernization project, and how many restrooms located in Concourse D are included in Phase 2?"

class MIAPhase2Extraction(BaseModel):
    scheduled_completion: Optional[str] = None
    scheduled_sources: List[str] = Field(default_factory=list)
    concourse_d_restroom_count: Optional[str] = None
    concourse_d_sources: List[str] = Field(default_factory=list)

def prompt_extract_mia_phase2() -> str:
    return """
    Extract the specific details for Miami International Airport's Restroom Modernization Project Phase 2 from the provided answer.
    Return a JSON object with the following fields:
    1. scheduled_completion: The scheduled completion date or timeframe for Phase 2 (e.g., "Q4 2025", "Summer 2025", "December 2025"). Extract exactly what the answer states.
    2. scheduled_sources: An array of URLs explicitly cited in the answer that support the Phase 2 scheduled completion date/timeframe. Include only actual URLs present in the answer.
    3. concourse_d_restroom_count: The number of restrooms (or sets of restrooms) located in Concourse D that are included in Phase 2. Extract exactly as stated (numbers may be numeric like "6" or words like "six").
    4. concourse_d_sources: An array of URLs explicitly cited in the answer that support the Concourse D restroom count included in Phase 2. Include only actual URLs present in the answer.

    Rules:
    - Do not invent or infer anything; extract exactly what appears in the answer text.
    - If any field is missing from the answer, set it to null (for strings) or an empty array (for sources).
    - For sources, only include valid URLs that are explicitly present in the answer (plain URLs or markdown links).
    """

async def build_mia_phase2_tree(
    evaluator: Evaluator,
    parent_node,
    extracted: MIAPhase2Extraction,
) -> None:
    mia_info_node = evaluator.add_parallel(
        id="MIA_Phase_2_Restroom_Modernization_Information",
        desc="Provides complete information about Miami International Airport's Phase 2 restroom modernization project, including both the scheduled completion timeframe and the scope of Concourse D restrooms",
        parent=parent_node,
        critical=True,
    )

    scheduled_node = evaluator.add_sequential(
        id="Scheduled_Completion_Date",
        desc="Provides the scheduled completion date or timeframe for Phase 2 of the restroom modernization project",
        parent=mia_info_node,
        critical=True,
    )

    scheduled_exists = evaluator.add_custom_node(
        result=bool(extracted.scheduled_completion and extracted.scheduled_completion.strip()) and len(extracted.scheduled_sources) > 0,
        id="Scheduled_Completion_Date_exists",
        desc="Scheduled completion date/timeframe is provided and has at least one cited source",
        parent=scheduled_node,
        critical=True,
    )

    scheduled_verify_leaf = evaluator.add_leaf(
        id="Scheduled_Completion_Date_supported",
        desc="Phase 2 scheduled completion date/timeframe is supported by cited source(s)",
        parent=scheduled_node,
        critical=True,
    )

    scheduled_claim = f"Phase 2 of Miami International Airport's restroom modernization is scheduled to be completed by {extracted.scheduled_completion}."
    await evaluator.verify(
        claim=scheduled_claim,
        node=scheduled_verify_leaf,
        sources=extracted.scheduled_sources,
        additional_instruction="Confirm the webpage explicitly states the planned/scheduled/expected completion for Phase 2 and that it matches the provided timeframe. Accept reasonable phrasing variants (e.g., 'expected', 'planned', 'target'). Ensure it refers to Phase 2, not Phase 1 or the overall project.",
    )

    concourse_node = evaluator.add_sequential(
        id="Concourse_D_Restroom_Count",
        desc="Provides the number of restrooms in Concourse D that are included in Phase 2",
        parent=mia_info_node,
        critical=True,
    )

    concourse_exists = evaluator.add_custom_node(
        result=bool(extracted.concourse_d_restroom_count and extracted.concourse_d_restroom_count.strip()) and len(extracted.concourse_d_sources) > 0,
        id="Concourse_D_Restroom_Count_exists",
        desc="Concourse D restroom count for Phase 2 is provided and has at least one cited source",
        parent=concourse_node,
        critical=True,
    )

    concourse_verify_leaf = evaluator.add_leaf(
        id="Concourse_D_Restroom_Count_supported",
        desc="Concourse D restroom count included in Phase 2 is supported by cited source(s)",
        parent=concourse_node,
        critical=True,
    )

    concourse_claim = f"Phase 2 includes {extracted.concourse_d_restroom_count} restrooms located in Concourse D at Miami International Airport."
    await evaluator.verify(
        claim=concourse_claim,
        node=concourse_verify_leaf,
        sources=extracted.concourse_d_sources,
        additional_instruction="Verify that the page explicitly states how many restrooms (or sets of restrooms) in Concourse D are included in Phase 2. Accept minor numeric format differences (e.g., 'six' vs '6'). Ensure the count is specifically tied to Phase 2 and Concourse D.",
    )

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

    extracted = await evaluator.extract(
        prompt=prompt_extract_mia_phase2(),
        template_class=MIAPhase2Extraction,
        extraction_name="mia_phase_2_extraction",
    )

    await build_mia_phase2_tree(evaluator, root, extracted)

    return evaluator.get_summary()