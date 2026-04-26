import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


TASK_ID = "ps5_pro_specs_2024"
TASK_DESCRIPTION = "What are the built-in SSD storage capacity, GPU performance in teraflops, and WiFi technology specification for the Sony PlayStation 5 Pro gaming console released in 2024?"


class PS5ProSpecs(BaseModel):
    device_name: Optional[str] = None
    release_year_mentioned: Optional[str] = None
    ssd_built_in_capacity: Optional[str] = None
    gpu_tflops: Optional[str] = None
    wifi_spec: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


def prompt_extract_ps5_pro_specs() -> str:
    return """
    Extract the technical specifications for the PlayStation 5 Pro as explicitly stated in the answer.
    Return a JSON object with the following fields:
    - device_name: The device name or variant exactly as mentioned (e.g., "PlayStation 5 Pro", "PS5 Pro").
    - release_year_mentioned: The year associated with this model as stated (e.g., "2024"). If year is not given, return null.
    - ssd_built_in_capacity: The built-in SSD storage capacity as mentioned in the answer (e.g., "2 TB", "2TB", "2000 GB").
    - gpu_tflops: The GPU performance in TFLOPS as stated (e.g., "33.5 TFLOPS").
    - wifi_spec: The Wi‑Fi technology/standard mentioned (e.g., "Wi‑Fi 7", "WiFi 7", "IEEE 802.11be").
    - sources: A list of all URLs explicitly cited in the answer that relate to these specs. Include only actual URLs present in the text (plain or in markdown).
    If any field is missing from the answer, return null for that field. If no URLs are present, return an empty list for sources.
    """


async def build_verification_tree(evaluator: Evaluator, specs: PS5ProSpecs) -> None:
    parent_node = evaluator.add_parallel(
        id="PS5_Pro_Technical_Specifications",
        desc="Verify the answer gives the requested technical specifications for the Sony PlayStation 5 Pro (2024 release).",
        parent=evaluator.root,
        critical=True
    )

    sources_list = specs.sources if specs and specs.sources else []

    context_leaf = evaluator.add_leaf(
        id="Correct_Entity_Context",
        desc="Answer clearly indicates the specs are for the Sony PlayStation 5 Pro and places it as the 2024 release (i.e., not the base PS5/PS5 Slim).",
        parent=parent_node,
        critical=True
    )
    context_claim = (
        "The answer clearly indicates that the specifications refer to the Sony PlayStation 5 Pro (PS5 Pro), "
        "specifically the 2024 release model, and not the base PS5 or the PS5 Slim."
    )
    context_add_ins = (
        f"Device name extracted: {specs.device_name or 'None'}; "
        f"Year extracted: {specs.release_year_mentioned or 'None'}. "
        "Consider common variants such as 'PS5 Pro' or 'PlayStation 5 Pro'. "
        "To pass, the answer must make it clear the model is the 2024 PS5 Pro, not base PS5 or PS5 Slim."
    )

    ssd_leaf = evaluator.add_leaf(
        id="Built_in_SSD_Storage_2TB",
        desc="Answer states the PS5 Pro includes 2TB of built-in SSD storage (2 TB / 2000 GB acceptable unit expression).",
        parent=parent_node,
        critical=True
    )
    ssd_claim = (
        "The answer states that the PS5 Pro includes 2 TB (two terabytes) of built-in SSD storage. "
        "Equivalent expressions such as '2TB' or '2000 GB' should be considered a match."
    )
    ssd_add_ins = (
        f"SSD capacity extracted: {specs.ssd_built_in_capacity or 'None'}. "
        "Accept '2 TB', '2TB', '2 terabytes', '2000 GB', or '2,000 GB' as correct. "
        "If the answer claims a different capacity (e.g., 1 TB) or omits capacity, it should fail."
    )

    gpu_leaf = evaluator.add_leaf(
        id="GPU_Performance_33_5_TFLOPS",
        desc="Answer states the PS5 Pro GPU performance is 33.5 teraflops (33.5 TFLOPS).",
        parent=parent_node,
        critical=True
    )
    gpu_claim = (
        "The answer states that the PS5 Pro's GPU performance is 33.5 TFLOPS (teraflops). "
        "Minor formatting variations like '33.5 TFlops' or 'approximately 33.5 TFLOPS' are acceptable."
    )
    gpu_add_ins = (
        f"GPU TFLOPS extracted: {specs.gpu_tflops or 'None'}. "
        "Accept small textual variants, but the numeric value must be 33.5."
    )

    wifi_leaf = evaluator.add_leaf(
        id="WiFi_7_Support",
        desc="Answer specifies the PS5 Pro supports Wi‑Fi 7.",
        parent=parent_node,
        critical=True
    )
    wifi_claim = (
        "The answer specifies that the PS5 Pro supports Wi‑Fi 7. "
        "Equivalent naming such as 'WiFi 7' or the standard 'IEEE 802.11be' should be considered a match."
    )
    wifi_add_ins = (
        f"Wi‑Fi spec extracted: {specs.wifi_spec or 'None'}. "
        "Accept 'Wi‑Fi 7', 'WiFi 7', or 'IEEE 802.11be' as correct."
    )

    claims_and_sources = [
        (context_claim, sources_list, context_leaf, context_add_ins),
        (ssd_claim, sources_list, ssd_leaf, ssd_add_ins),
        (gpu_claim, sources_list, gpu_leaf, gpu_add_ins),
        (wifi_claim, sources_list, wifi_leaf, wifi_add_ins),
    ]
    await evaluator.batch_verify(claims_and_sources)


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

    specs = await evaluator.extract(
        prompt=prompt_extract_ps5_pro_specs(),
        template_class=PS5ProSpecs,
        extraction_name="ps5_pro_specs"
    )

    evaluator.add_ground_truth({
        "expected_device": "Sony PlayStation 5 Pro (2024)",
        "expected_specs": {
            "ssd_built_in_capacity": "2 TB",
            "gpu_tflops": "33.5 TFLOPS",
            "wifi_spec": "Wi‑Fi 7 (IEEE 802.11be)"
        }
    }, gt_type="ground_truth")

    await build_verification_tree(evaluator, specs)

    return evaluator.get_summary()