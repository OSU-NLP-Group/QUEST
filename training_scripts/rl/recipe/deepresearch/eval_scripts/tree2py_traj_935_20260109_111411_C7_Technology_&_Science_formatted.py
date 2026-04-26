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
TASK_ID = "edge_object_detection_models_eval"
TASK_DESCRIPTION = """
I am developing a real-time object detection application for deployment on edge devices with commercial usage. I need to identify object detection models that meet specific technical requirements for both performance and licensing.

Find 4 different object detection models that satisfy ALL of the following criteria:

1. The model must be released under a permissive open-source license (Apache 2.0, MIT, or equivalent) that allows commercial use without copyleft restrictions.

2. The model must achieve at least 50.0 mAP (mean Average Precision) on the COCO validation dataset.

3. The model must achieve at least 50 FPS (frames per second) when benchmarked on an NVIDIA T4 GPU with TensorRT FP16 optimization.

4. The model must have 50 million parameters or fewer.

For each of the 4 models, provide:
- The model name (including version/variant if applicable)
- The specific license under which it is released
- The exact mAP score achieved on COCO validation set
- The exact FPS achieved on NVIDIA T4 GPU with TensorRT FP16
- The total number of parameters (in millions)
- A direct URL to the model's official repository, model card, or documentation page where these specifications are documented
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ModelEntry(BaseModel):
    model_name: Optional[str] = None
    license: Optional[str] = None
    map_coco_val: Optional[str] = None
    fps_t4_trt_fp16: Optional[str] = None
    params_million: Optional[str] = None
    official_urls: List[str] = Field(default_factory=list)
    backbone: Optional[str] = None
    pretrained_weights_available: Optional[bool] = None


class ModelsExtraction(BaseModel):
    models: List[ModelEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_models() -> str:
    return """
    Extract up to 8 object detection models mentioned in the answer that are claimed to meet the user's requirements.
    For each model, extract the following fields exactly as presented in the answer (do not infer new values):
    - model_name: The model name including version/variant if mentioned (e.g., "YOLOv5s", "YOLOX-Tiny", "RT-DETR-R50").
    - license: The specific open-source license string as stated (e.g., "Apache-2.0", "MIT", "BSD-3-Clause", etc.). If multiple are shown, pick the applicable one for the model code/weights.
    - map_coco_val: The exact mAP value (bbox/AP@[.5:.95]) on the COCO validation set, as written (e.g., "51.2" or "51.2 mAP"). Prefer the standard COCO metric.
    - fps_t4_trt_fp16: The exact FPS measured on an NVIDIA T4 GPU using TensorRT FP16 as written (e.g., "52", "54.3 FPS", ">= 50"). Keep the exact value/string from the answer.
    - params_million: The total number of parameters in millions as written (e.g., "7.5M", "7.5", "7.5 million"). Keep the value string.
    - official_urls: A list of direct URLs that are official repository/model card/documentation pages for the model. Only include URLs that are actually present in the answer. Do not fabricate URLs.
    - backbone: The backbone architecture string if specified (e.g., "CSPDarknet", "ResNet-50", "Swin-T", "MobileNetV3"). If none, set to null.
    - pretrained_weights_available: true if the answer explicitly indicates pretrained weights/checkpoints are publicly available (e.g., download links on the official page), otherwise false or null.

    Important:
    - Extract only values explicitly present in the answer text. If a value is missing, set it to null (or false/null for boolean).
    - For official_urls, include only valid URLs. If present as markdown links, extract the actual URL.
    - Do not attempt to normalize or convert numbers; keep the exact strings from the answer.
    - Return a JSON object with a "models" array, each element respecting the schema.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def normalize_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def nonempty_str(x: Optional[str]) -> bool:
    return isinstance(x, str) and x.strip() != ""


# --------------------------------------------------------------------------- #
# Verification builder per model                                              #
# --------------------------------------------------------------------------- #
async def build_and_verify_model(
    evaluator: Evaluator,
    root_node,
    model: ModelEntry,
    index: int,
) -> None:
    idx1 = index + 1
    desc = f"Evaluation of the {ordinal(idx1)} model against all required fields and constraints"
    model_node = evaluator.add_parallel(
        id=f"model_{index+1}",
        desc=desc,
        parent=root_node,
        critical=False,  # allow partial credit across models
    )

    # 1) Name_and_variant (critical) — ensure a name is provided
    evaluator.add_custom_node(
        result=nonempty_str(model.model_name),
        id=f"model_{index+1}_name_and_variant",
        desc="Provides the model name (including version/variant if applicable)",
        parent=model_node,
        critical=True,
    )

    # 2) Official_URL_with_specs (critical) — verify at least one official URL is provided and is an official page
    official_url_leaf = evaluator.add_leaf(
        id=f"model_{index+1}_official_url_with_specs",
        desc="Provides a direct URL to an official repository/model card/documentation page that is official for the model",
        parent=model_node,
        critical=True,
    )
    url_claim = f"This page is an official repository, model card, or documentation page for the model '{model.model_name or 'UNKNOWN'}'."
    await evaluator.verify(
        claim=url_claim,
        node=official_url_leaf,
        sources=normalize_urls(model.official_urls),
        additional_instruction=(
            "Treat as official if the page is a repository or documentation maintained by the original authors or "
            "organization (e.g., GitHub org, official docs site, official HuggingFace model card). "
            "If the provided URL(s) are third-party blogs or unrelated pages, mark as Incorrect."
        ),
    )

    # Prepare a shared sources list and precondition
    sources = normalize_urls(model.official_urls)
    prerequisites = [official_url_leaf]

    # 3) License_permissive_commercial (critical) — verify license and permissiveness
    license_leaf = evaluator.add_leaf(
        id=f"model_{index+1}_license_permissive",
        desc="States the specific license and it is permissive (e.g., Apache-2.0, MIT, or equivalent) allowing commercial use without copyleft restrictions",
        parent=model_node,
        critical=True,
    )
    license_str = model.license or "UNKNOWN"
    license_claim = (
        f"The model '{model.model_name or 'UNKNOWN'}' is released under the '{license_str}' license, "
        "which is a permissive open-source license that allows commercial use without copyleft restrictions (e.g., Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, ISC, or Unlicense)."
    )
    await evaluator.verify(
        claim=license_claim,
        node=license_leaf,
        sources=sources,
        additional_instruction=(
            "Verify that the page explicitly states the given license string. "
            "Accept permissive licenses such as Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, ISC, or Unlicense. "
            "Reject copyleft licenses like GPL/AGPL/LGPL. If license isn't shown, mark as Incorrect."
        ),
        extra_prerequisites=prerequisites,
    )

    # 4) mAP_COCO_val_at_least_50_and_reported (critical)
    map_leaf = evaluator.add_leaf(
        id=f"model_{index+1}_map_coco_val_ge50",
        desc="Reports an exact mAP value for COCO validation using standard COCO evaluation protocol, and the value is ≥ 50.0",
        parent=model_node,
        critical=True,
    )
    map_str = model.map_coco_val or "UNKNOWN"
    map_claim = (
        f"On the COCO validation dataset (bbox AP, standard COCO evaluation), the model '{model.model_name or 'UNKNOWN'}' "
        f"achieves an mAP value of {map_str}, which is at least 50.0."
    )
    await evaluator.verify(
        claim=map_claim,
        node=map_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the page shows an exact bbox mAP (AP@[.5:.95] or 'box mAP') for COCO validation. "
            "The value must be ≥ 50.0 when expressed on a 0–100 scale. "
            "If mAP is not reported for COCO val, or is < 50.0, mark as Incorrect."
        ),
        extra_prerequisites=prerequisites,
    )

    # 5) FPS_T4_TensorRT_FP16_at_least_50_and_reported (critical)
    fps_leaf = evaluator.add_leaf(
        id=f"model_{index+1}_fps_t4_trt_fp16_ge50",
        desc="Reports an exact FPS measured on NVIDIA T4 with TensorRT FP16, and the value is ≥ 50 FPS",
        parent=model_node,
        critical=True,
    )
    fps_str = model.fps_t4_trt_fp16 or "UNKNOWN"
    fps_claim = (
        f"When benchmarked on an NVIDIA T4 GPU with TensorRT FP16, the model '{model.model_name or 'UNKNOWN'}' "
        f"achieves {fps_str} FPS, which is at least 50 FPS."
    )
    await evaluator.verify(
        claim=fps_claim,
        node=fps_leaf,
        sources=sources,
        additional_instruction=(
            "The page must explicitly specify the throughput measured on an NVIDIA T4 GPU using TensorRT with FP16. "
            "If the benchmark is on a different GPU or without TensorRT FP16, mark as Incorrect. "
            "The reported FPS must be ≥ 50."
        ),
        extra_prerequisites=prerequisites,
    )

    # 6) Parameters_at_most_50M_and_reported (critical)
    params_leaf = evaluator.add_leaf(
        id=f"model_{index+1}_params_le_50m",
        desc="Reports the parameter count (in millions) and it is ≤ 50M",
        parent=model_node,
        critical=True,
    )
    params_str = model.params_million or "UNKNOWN"
    params_claim = (
        f"The model '{model.model_name or 'UNKNOWN'}' has a total of {params_str} parameters in millions, which is at most 50 million parameters."
    )
    await evaluator.verify(
        claim=params_claim,
        node=params_leaf,
        sources=sources,
        additional_instruction=(
            "Verify the parameter count on the page. If shown in 'M' or 'million', interpret accordingly. "
            "The count must be ≤ 50 million. If no parameter count is provided, mark as Incorrect."
        ),
        extra_prerequisites=prerequisites,
    )

    # 7) Pretrained_weights_publicly_available (critical)
    weights_leaf = evaluator.add_leaf(
        id=f"model_{index+1}_pretrained_weights_available",
        desc="Indicates pretrained weights are publicly available (as documented by an official source)",
        parent=model_node,
        critical=True,
    )
    weights_claim = (
        f"Pretrained weights for the model '{model.model_name or 'UNKNOWN'}' are publicly available for download (e.g., checkpoints or weights linked on the official page)."
    )
    await evaluator.verify(
        claim=weights_claim,
        node=weights_leaf,
        sources=sources,
        additional_instruction=(
            "Look for explicit indications of downloadable pretrained checkpoints/weights (e.g., .pt, .onnx, .engine, or links to HF model weights). "
            "If there is no clear mention of publicly available pretrained weights, mark as Incorrect."
        ),
        extra_prerequisites=prerequisites,
    )

    # 8) Backbone_specified (critical)
    backbone_leaf = evaluator.add_leaf(
        id=f"model_{index+1}_backbone_specified",
        desc="Backbone architecture is clearly specified (as documented by an official source)",
        parent=model_node,
        critical=True,
    )
    backbone_str = model.backbone or "UNKNOWN"
    backbone_claim = (
        f"The model '{model.model_name or 'UNKNOWN'}' uses the backbone '{backbone_str}'."
    )
    await evaluator.verify(
        claim=backbone_claim,
        node=backbone_leaf,
        sources=sources,
        additional_instruction=(
            "Verify that the page explicitly names the backbone architecture (e.g., CSPDarknet, ResNet-50, Swin-T, MobileNetV3). "
            "If the backbone is not clearly specified, mark as Incorrect."
        ),
        extra_prerequisites=prerequisites,
    )


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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # models can be evaluated independently
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

    # Extract structured model info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_models(),
        template_class=ModelsExtraction,
        extraction_name="extracted_models",
    )

    # Normalize to exactly 4 entries (truncate or pad with empty)
    models: List[ModelEntry] = list(extracted.models[:4])
    while len(models) < 4:
        models.append(ModelEntry())

    # Critical check: Provides exactly 4 distinct models (by non-empty names)
    names = [m.model_name.strip().lower() for m in models if nonempty_str(m.model_name)]
    distinct_ok = (len(names) == 4) and (len(set(names)) == 4)

    evaluator.add_custom_node(
        result=distinct_ok,
        id="provides_4_distinct_models",
        desc="Provides exactly 4 models and they are distinct (not the same model repeated under different labels)",
        parent=root,
        critical=True,  # essential gating criterion per rubric
    )

    # Build and verify each model
    for i, m in enumerate(models):
        await build_and_verify_model(evaluator, root, m, i)

    # Return evaluation summary
    return evaluator.get_summary()