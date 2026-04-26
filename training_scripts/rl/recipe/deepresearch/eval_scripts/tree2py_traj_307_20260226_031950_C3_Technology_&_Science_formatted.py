import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nvidia_dlss45_rtx30_series_recommendation"
TASK_DESCRIPTION = """
NVIDIA released DLSS 4.5 Super Resolution in January 2026, introducing three model presets through the NVIDIA app:
- Model M (optimized for Performance mode),
- Model L (optimized for 4K Ultra Performance mode),
- Model K (DLSS 4).

A PC gamer owns a GeForce RTX 30 Series graphics card and wants to enable DLSS 4.5 for optimal gaming performance.
According to NVIDIA's official guidance, which model preset should this gamer select, and what is the primary technical limitation of RTX 30 Series GPUs that influences this recommendation?
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class DLSS45Extraction(BaseModel):
    """
    Structured extraction from the agent's answer.
    """
    recommended_preset: Optional[str] = None  # Expected to be "Model K", "DLSS 4", or equivalent phrasing
    primary_limitation: Optional[str] = None  # Expected to mention lack of native FP8 support
    sources: List[str] = Field(default_factory=list)  # Any URLs cited in the answer (prefer NVIDIA official)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_dlss45() -> str:
    return """
    From the answer, extract:
    1) recommended_preset: The model preset the answer says an RTX 30 Series gamer should select for DLSS 4.5 Super Resolution in the NVIDIA app.
       - Normalize to a concise string as it appears in the answer.
       - Accept typical variants/synonyms like "Model K", "K", "DLSS 4", "Model K (DLSS 4)". Do not invent new terms.
       - If not explicitly stated, return null.

    2) primary_limitation: The primary technical limitation of RTX 30 Series cited in the answer that drives this recommendation.
       - Return the phrase exactly as it appears (e.g., "lack of native FP8 support", "no FP8 tensor-core support").
       - If not explicitly stated, return null.

    3) sources: All URLs cited in the answer that support this recommendation and/or limitation.
       - Extract actual URLs only (including those inside markdown links).
       - Prefer NVIDIA official domains if present (e.g., nvidia.com, developer.nvidia.com).
       - If none are present, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, extracted: DLSS45Extraction) -> None:
    """
    Build the verification tree based on the rubric and run verifications.
    The rubric requires two critical checks under a critical, parallel parent:
    1) The recommended model preset (Model K / DLSS 4).
    2) The primary technical limitation (lack of native FP8 support).
    """

    # Critical parent (parallel aggregation)
    complete_node = evaluator.add_parallel(
        id="complete_answer_evaluation",
        desc="Evaluates whether the answer identifies NVIDIA's recommended model preset for RTX 30 Series and the key hardware limitation driving that recommendation.",
        parent=evaluator.root,
        critical=True
    )

    # 1) Recommended model preset (critical leaf)
    preset_leaf = evaluator.add_leaf(
        id="recommended_model_preset",
        desc="Answer identifies the recommended model preset for GeForce RTX 30 Series GPUs (Model K / DLSS 4).",
        parent=complete_node,
        critical=True
    )

    # We verify whether, in the answer text, the recommended preset is Model K (i.e., DLSS 4).
    # This is a straightforward content check over the answer (simple_verify).
    recommended_preset_claim = (
        "In the provided answer, the recommended DLSS 4.5 Super Resolution model preset for "
        "GeForce RTX 30 Series GPUs is 'Model K' (also referred to as DLSS 4)."
    )
    await evaluator.verify(
        claim=recommended_preset_claim,
        node=preset_leaf,
        sources=None,  # This is a direct check against the answer text
        additional_instruction=(
            "Evaluate solely based on the answer text. Consider the following as equivalent to 'Model K': "
            "'Model K', 'K', 'DLSS 4', 'Model K (DLSS 4)'. "
            "If the answer recommends 'Model M' or 'Model L' for RTX 30 Series as the main DLSS 4.5 preset, mark it incorrect."
        )
    )

    # 2) Primary technical limitation (critical leaf)
    limitation_leaf = evaluator.add_leaf(
        id="primary_technical_limitation",
        desc="Answer states the primary technical limitation of RTX 30 Series that drives the recommendation: lack of native FP8 support.",
        parent=complete_node,
        critical=True
    )

    # We verify that the answer attributes the recommendation to a lack of native FP8 support on RTX 30 (Ampere) GPUs.
    limitation_claim = (
        "In the provided answer, the primary technical limitation cited for RTX 30 Series GPUs that drives the "
        "DLSS 4.5 preset recommendation is the lack of native FP8 support (e.g., no FP8 on the Tensor Cores)."
    )
    await evaluator.verify(
        claim=limitation_claim,
        node=limitation_leaf,
        sources=None,  # Directly check the answer text for this specific rationale
        additional_instruction=(
            "Evaluate solely based on the answer text. Accept equivalent phrasing such as: "
            "'no FP8 support', 'lack of FP8', 'no 8-bit floating point (FP8) support', 'no FP8 tensor-core support'. "
            "If the answer primarily cites a different factor instead of FP8—for example, only mentions generic performance, "
            "Optical Flow version, or non-FP8 reasons—mark it incorrect."
        )
    )

    # Optionally record what we extracted and our ground-truth expectations for transparency
    evaluator.add_ground_truth({
        "expected_recommended_preset": "Model K (DLSS 4)",
        "expected_primary_limitation": "Lack of native FP8 support on RTX 30 Series (Ampere) Tensor Cores."
    }, gt_type="expected_answer")


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
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
    """
    Entry point for evaluating an answer against the DLSS 4.5 / RTX 30 Series recommendation rubric.
    """
    # Initialize evaluator with a parallel root
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_dlss45(),
        template_class=DLSS45Extraction,
        extraction_name="dlss45_extraction"
    )

    # Build verification tree and run checks
    await build_and_verify(evaluator, extracted)

    # Return standardized summary
    return evaluator.get_summary()