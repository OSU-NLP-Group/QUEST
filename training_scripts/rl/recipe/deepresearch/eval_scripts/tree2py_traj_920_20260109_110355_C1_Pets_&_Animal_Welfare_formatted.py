import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "la_dacc_microchip_fee"
TASK_DESCRIPTION = "What is the microchip fee charged by the Los Angeles County Department of Animal Care and Control (LA County DACC) when adopting a pet, and does this fee include the national microchip registry registration?"


class MicrochipFeeExtraction(BaseModel):
    microchip_fee_amount: Optional[str] = None
    registration_included: Optional[str] = None  # expected values: "included", "not_included"
    source_urls: List[str] = Field(default_factory=list)


def prompt_extract_microchip_info() -> str:
    return """
    Extract the microchip fee information provided in the answer concerning the Los Angeles County Department of Animal Care and Control (LA County DACC) for pet adoptions.

    You must extract:
    1) microchip_fee_amount: The specific fee amount mentioned for microchipping during LA County DACC pet adoption (include currency symbol if present, e.g., "$20"). If multiple amounts are mentioned, pick the one explicitly tied to LA County DACC microchip fee when adopting a pet. If not stated, return null.
    2) registration_included: Whether the national microchip registry registration fee is included in the stated microchip fee. Return:
       - "included" if the answer clearly states inclusion,
       - "not_included" if the answer clearly states it is not included or requires a separate fee,
       - null if it is not stated or unclear.
    3) source_urls: All explicit URLs cited in the answer that are related to LA County DACC microchip/adoption fees or microchip registration. Include only valid URLs. If none are cited, return an empty list.

    Notes:
    - Do not invent information.
    - Only extract what is explicitly in the answer text.
    - For registration_included, look for clear phrases like "includes national registry registration" or "registration is separate/not included".
    """


async def build_tree_and_verify(evaluator: Evaluator, extracted: MicrochipFeeExtraction) -> None:
    main_node = evaluator.add_parallel(
        id="Microchip_Fee_Information",
        desc="The answer provides the microchip fee charged by LA County DACC when adopting a pet and states whether national microchip registry registration is included.",
        parent=evaluator.root,
        critical=True
    )

    fee_leaf = evaluator.add_leaf(
        id="Fee_Amount",
        desc="The answer provides the specific microchip fee amount charged by LA County DACC.",
        parent=main_node,
        critical=True
    )

    reg_leaf = evaluator.add_leaf(
        id="Registration_Clarification",
        desc="The answer clarifies whether the national microchip registry registration fee is included in the stated microchip fee.",
        parent=main_node,
        critical=True
    )

    if extracted.microchip_fee_amount and extracted.microchip_fee_amount.strip():
        fee_claim = f"The microchip fee charged by the Los Angeles County Department of Animal Care and Control when adopting a pet is {extracted.microchip_fee_amount}."
        await evaluator.verify(
            claim=fee_claim,
            node=fee_leaf,
            sources=extracted.source_urls if extracted.source_urls else None,
            additional_instruction="Verify that the answer explicitly provides a specific microchip fee amount for LA County DACC adoptions. If source URLs are provided, confirm the fee amount is supported by the LA County DACC or directly relevant official/adoption pages."
        )
    else:
        fee_leaf.score = 0.0
        fee_leaf.status = "failed"

    if extracted.registration_included in ("included", "not_included"):
        reg_text = (
            "The national microchip registry registration fee is included in the microchip fee."
            if extracted.registration_included == "included"
            else "The national microchip registry registration fee is not included in the microchip fee."
        )
        await evaluator.verify(
            claim=reg_text,
            node=reg_leaf,
            sources=extracted.source_urls if extracted.source_urls else None,
            additional_instruction="Verify that the answer clearly states whether the national microchip registry registration fee is included in, or excluded from, the stated microchip fee. If source URLs are provided, confirm the inclusion/exclusion claim is supported by the LA County DACC or directly relevant official/adoption pages."
        )
    else:
        reg_leaf.score = 0.0
        reg_leaf.status = "failed"


async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_microchip_info(),
        template_class=MicrochipFeeExtraction,
        extraction_name="microchip_fee_info"
    )

    await build_tree_and_verify(evaluator, extracted)

    return evaluator.get_summary()