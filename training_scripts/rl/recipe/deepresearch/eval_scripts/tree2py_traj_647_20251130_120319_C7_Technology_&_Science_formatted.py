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
TASK_ID = "console_compare_ps5pro_xboxsx"
TASK_DESCRIPTION = (
    "A gaming enthusiast in Texas is comparing the PlayStation 5 Pro and Xbox Series X to decide which console to purchase "
    "from local retailers such as Best Buy, GameStop, Walmart, or Target. Provide a comprehensive technical comparison that "
    "includes the following specifications for both consoles: internal storage capacity, GPU performance (in teraflops), "
    "ray tracing capabilities, maximum frame rate support, maximum resolution support, and backward compatibility features. "
    "Additionally, include the release date and retail price (in USD) for the PS5 Pro."
)

# Ground-truth targets as specified by rubric requirements (for record-keeping/debugging)
EXPECTED_SPECS = {
    "ps5_pro": {
        "storage": "2TB SSD",
        "gpu_teraflops": "16.7 teraflops",
        "ray_tracing": "supports advanced ray tracing",
        "max_frame_rate": "up to 120 frames per second",
        "max_resolution": "4K resolution gaming",
        "release_date": "November 7, 2024",
        "price_usd": "$699.99 USD",
        "backward_compatibility": "backward compatible with PS4 games",
    },
    "xbox_series_x": {
        "storage": "1TB SSD",
        "gpu_teraflops": "12 teraflops",
        "ray_tracing": "supports hardware-accelerated ray tracing",
        "max_frame_rate": "up to 120 frames per second",
        "max_resolution": "4K resolution gaming",
        "backward_compatibility": "backward compatible with previous Xbox generation games",
    }
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ConsoleSpecs(BaseModel):
    name: Optional[str] = None

    storage: Optional[str] = None
    storage_sources: List[str] = Field(default_factory=list)

    gpu_teraflops: Optional[str] = None
    gpu_sources: List[str] = Field(default_factory=list)

    ray_tracing: Optional[str] = None
    ray_tracing_sources: List[str] = Field(default_factory=list)

    max_frame_rate: Optional[str] = None
    frame_rate_sources: List[str] = Field(default_factory=list)

    max_resolution: Optional[str] = None
    resolution_sources: List[str] = Field(default_factory=list)

    backward_compatibility: Optional[str] = None
    backward_compatibility_sources: List[str] = Field(default_factory=list)

    # PS5 Pro specific fields (will be null for Xbox Series X)
    release_date: Optional[str] = None
    release_date_sources: List[str] = Field(default_factory=list)

    price_usd: Optional[str] = None
    price_sources: List[str] = Field(default_factory=list)

    # General catch-all sources cited for this console, if any
    general_sources: List[str] = Field(default_factory=list)


class ConsoleComparisonExtraction(BaseModel):
    ps5_pro: Optional[ConsoleSpecs] = None
    xbox_series_x: Optional[ConsoleSpecs] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_console_specs() -> str:
    return """
    Extract the technical specifications for the PlayStation 5 Pro (PS5 Pro) and Xbox Series X as they are presented in the answer text.
    Your goal is to structure the answer’s content into the following fields for each console. Additionally, extract the specific URLs (if any) cited in the answer that support each field.

    For each console (PS5 Pro and Xbox Series X), extract:
    - name: The console name used in the answer (e.g., "PlayStation 5 Pro", "PS5 Pro", "Xbox Series X").
    - storage: The internal storage capacity (e.g., "2TB SSD", "1 TB SSD", "1TB NVMe SSD").
    - storage_sources: All URLs that the answer associates with the storage specification.
    - gpu_teraflops: The GPU performance in teraflops as stated (e.g., "16.7 TFLOPS", "12 teraflops").
    - gpu_sources: All URLs associated with the GPU specification.
    - ray_tracing: A short phrase describing ray tracing support (e.g., "supports advanced ray tracing", "hardware-accelerated ray tracing").
    - ray_tracing_sources: All URLs associated with ray tracing details.
    - max_frame_rate: The maximum supported frame rate as stated (e.g., "up to 120 FPS", "120 frames per second").
    - frame_rate_sources: All URLs tied to the frame rate claim.
    - max_resolution: The maximum supported resolution as stated (e.g., "4K", "up to 4K").
    - resolution_sources: All URLs tied to the resolution claim.
    - backward_compatibility: A short phrase describing backward compatibility (e.g., "backward compatible with PS4 games", "backward compatible with previous Xbox generations").
    - backward_compatibility_sources: All URLs associated with backward compatibility statements.
    - general_sources: Any general/spec overview URLs cited for the console that may support multiple specs.

    Additionally for PS5 Pro, extract:
    - release_date: The release date for PS5 Pro if stated (e.g., "November 7, 2024").
    - release_date_sources: All URLs specifically tied to the PS5 Pro release date.
    - price_usd: The retail price (in USD) for PS5 Pro if stated (e.g., "$699.99 USD", "699.99 USD").
    - price_sources: All URLs tied to the PS5 Pro price claim.

    IMPORTANT:
    - Only extract URLs that are explicitly present in the answer text (including markdown links). Do not infer or invent URLs.
    - Return null for any missing scalar field and an empty array for any missing list of URLs.
    - Preserve the phrasing found in the answer for text fields (do not normalize to your own wording).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def pick_sources(preferred: Optional[List[str]], fallback: Optional[List[str]]) -> Optional[List[str]]:
    """
    Pick sources for a verification: prefer a specific list; if empty, fall back to general.
    Return None if both are empty or None.
    """
    if preferred and len(preferred) > 0:
        return preferred
    if fallback and len(fallback) > 0:
        return fallback
    return None


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_comparison(
    evaluator: Evaluator,
    root,
    extracted: ConsoleComparisonExtraction
) -> None:
    """
    Build the verification tree based on the rubric and launch verifications.
    The rubric root is a parallel node with 14 critical leaf checks.
    """
    # Create rubric root as a child of global root
    comp_node = evaluator.add_parallel(
        id="gaming_console_comparison",
        desc="Comprehensive comparison of PS5 Pro and Xbox Series X technical specifications per the provided constraints",
        parent=root,
        critical=False
    )

    ps5 = extracted.ps5_pro or ConsoleSpecs()
    xbox = extracted.xbox_series_x or ConsoleSpecs()

    # Prepare leaf nodes and corresponding claims/sources
    claims_and_sources: List[
        tuple[str, Optional[List[str]] | Optional[str], Any, Optional[str]]
    ] = []

    # Helper to add a leaf and queue verification
    def add_leaf_and_queue(node_id: str, desc: str, claim: str, sources: Optional[List[str]], add_ins: str) -> None:
        node = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=comp_node,
            critical=True
        )
        claims_and_sources.append((claim, sources, node, add_ins))

    # Additional instruction templates
    ins_storage = "Allow minor formatting variants like '2TB' vs '2 TB' and mention of NVMe/SSD explicitly or implicitly. The claim is supported if the page clearly indicates this internal storage capacity."
    ins_gpu = "Focus on single-precision (FP32) TFLOPS unless the page explicitly clarifies otherwise. Allow minor rounding differences (e.g., 16.7 vs 16.70)."
    ins_rt_adv = "Support the claim if the page states ray tracing support and qualifies it as enhanced/advanced or improved ray tracing on PS5 Pro."
    ins_rt_hw = "Support the claim if the page states hardware-accelerated ray tracing for Xbox Series X (or equivalent phrasing)."
    ins_fps = "Support the claim if the page indicates the console supports up to 120 FPS (120 frames per second)."
    ins_res = "Support the claim if the page indicates the console supports 4K resolution gaming (phrases like 'up to 4K', 'native 4K', or similar are acceptable)."
    ins_date = "Support the claim if the page indicates the PS5 Pro release date as November 7, 2024. Allow minor date format variations (e.g., 'Nov 7, 2024', '2024-11-07')."
    ins_price = "Support the claim if the page clearly indicates the PS5 Pro retail price as $699.99 USD (allow format variants like 'USD 699.99' or '$699.99')."
    ins_bc_ps = "Support the claim if the page states PS5 Pro (as a PS5 family console) is backward compatible with PS4 games (phrases like 'plays most PS4 games' acceptable)."
    ins_bc_xbox = "Support the claim if the page states Xbox Series X is backward compatible with previous Xbox generations (e.g., Xbox One, Xbox 360, and Original Xbox where applicable)."

    # 1) PS5 Pro internal storage (2TB SSD)
    add_leaf_and_queue(
        "ps5_pro_storage",
        "States PS5 Pro internal storage capacity as 2TB SSD",
        "The PlayStation 5 Pro (PS5 Pro) internal storage capacity is 2 TB SSD.",
        pick_sources(ps5.storage_sources, ps5.general_sources),
        ins_storage
    )

    # 2) Xbox Series X internal storage (1TB SSD)
    add_leaf_and_queue(
        "xbox_series_x_storage",
        "States Xbox Series X internal storage capacity as 1TB SSD",
        "The Xbox Series X internal storage capacity is 1 TB SSD.",
        pick_sources(xbox.storage_sources, xbox.general_sources),
        ins_storage
    )

    # 3) PS5 Pro GPU performance (16.7 TFLOPS)
    add_leaf_and_queue(
        "ps5_pro_gpu",
        "States PS5 Pro GPU performance as 16.7 teraflops",
        "The PlayStation 5 Pro (PS5 Pro) GPU performance is 16.7 teraflops (TFLOPS).",
        pick_sources(ps5.gpu_sources, ps5.general_sources),
        ins_gpu
    )

    # 4) Xbox Series X GPU performance (12 TFLOPS)
    add_leaf_and_queue(
        "xbox_series_x_gpu",
        "States Xbox Series X GPU performance as 12 teraflops",
        "The Xbox Series X GPU performance is 12 teraflops (TFLOPS).",
        pick_sources(xbox.gpu_sources, xbox.general_sources),
        ins_gpu
    )

    # 5) PS5 Pro ray tracing (advanced)
    add_leaf_and_queue(
        "ps5_pro_ray_tracing",
        "States PS5 Pro supports advanced ray tracing technology",
        "The PlayStation 5 Pro (PS5 Pro) supports advanced ray tracing technology.",
        pick_sources(ps5.ray_tracing_sources, ps5.general_sources),
        ins_rt_adv
    )

    # 6) Xbox Series X ray tracing (hardware-accelerated)
    add_leaf_and_queue(
        "xbox_series_x_ray_tracing",
        "States Xbox Series X supports hardware-accelerated ray tracing",
        "The Xbox Series X supports hardware-accelerated ray tracing.",
        pick_sources(xbox.ray_tracing_sources, xbox.general_sources),
        ins_rt_hw
    )

    # 7) PS5 Pro max frame rate (up to 120 fps)
    add_leaf_and_queue(
        "ps5_pro_frame_rate",
        "States PS5 Pro supports gaming at up to 120 frames per second",
        "The PlayStation 5 Pro (PS5 Pro) supports gaming at up to 120 frames per second (120 FPS).",
        pick_sources(ps5.frame_rate_sources, ps5.general_sources),
        ins_fps
    )

    # 8) Xbox Series X max frame rate (up to 120 fps)
    add_leaf_and_queue(
        "xbox_series_x_frame_rate",
        "States Xbox Series X supports gaming at up to 120 frames per second",
        "The Xbox Series X supports gaming at up to 120 frames per second (120 FPS).",
        pick_sources(xbox.frame_rate_sources, xbox.general_sources),
        ins_fps
    )

    # 9) PS5 Pro max resolution (4K)
    add_leaf_and_queue(
        "ps5_pro_resolution",
        "States PS5 Pro supports 4K resolution gaming",
        "The PlayStation 5 Pro (PS5 Pro) supports 4K resolution gaming.",
        pick_sources(ps5.resolution_sources, ps5.general_sources),
        ins_res
    )

    # 10) Xbox Series X max resolution (4K)
    add_leaf_and_queue(
        "xbox_series_x_resolution",
        "States Xbox Series X supports 4K resolution gaming",
        "The Xbox Series X supports 4K resolution gaming.",
        pick_sources(xbox.resolution_sources, xbox.general_sources),
        ins_res
    )

    # 11) PS5 Pro release date (November 7, 2024)
    add_leaf_and_queue(
        "ps5_pro_release_date",
        "States PS5 Pro release date as November 7, 2024",
        "The PlayStation 5 Pro (PS5 Pro) release date is November 7, 2024.",
        pick_sources(ps5.release_date_sources, ps5.general_sources),
        ins_date
    )

    # 12) PS5 Pro price ($699.99 USD)
    add_leaf_and_queue(
        "ps5_pro_price",
        "States PS5 Pro retail price as $699.99 USD",
        "The PlayStation 5 Pro (PS5 Pro) retail price is $699.99 USD.",
        pick_sources(ps5.price_sources, ps5.general_sources),
        ins_price
    )

    # 13) PS5 Pro backward compatibility (PS4)
    add_leaf_and_queue(
        "ps5_pro_backward_compatibility",
        "States PS5 Pro is backward compatible with PS4 games",
        "The PlayStation 5 Pro (PS5 Pro) is backward compatible with PS4 games.",
        pick_sources(ps5.backward_compatibility_sources, ps5.general_sources),
        ins_bc_ps
    )

    # 14) Xbox Series X backward compatibility (previous Xbox generations)
    add_leaf_and_queue(
        "xbox_series_x_backward_compatibility",
        "States Xbox Series X is backward compatible with previous Xbox generation games",
        "The Xbox Series X is backward compatible with previous Xbox generation games.",
        pick_sources(xbox.backward_compatibility_sources, xbox.general_sources),
        ins_bc_xbox
    )

    # Run all verifications in parallel for efficiency
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the PS5 Pro vs Xbox Series X comparison task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # The rubric root aggregates children in parallel
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_console_specs(),
        template_class=ConsoleComparisonExtraction,
        extraction_name="console_spec_extraction"
    )

    # Add expected info (for transparency; not used directly in scoring)
    evaluator.add_ground_truth({
        "expected": EXPECTED_SPECS,
        "notes": "These are the rubric-expected statements for evaluation."
    })

    # Build and verify rubric tree
    await build_and_verify_comparison(evaluator, root, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()