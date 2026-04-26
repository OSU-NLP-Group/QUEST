import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "consumer_gpu_blackwell_comprehensive_specs"
TASK_DESCRIPTION = (
    "I am researching high-performance consumer graphics cards for AI workload acceleration and need to identify a specific GPU model "
    "that meets a comprehensive set of technical requirements. The GPU must satisfy ALL of the following constraints:\n\n"
    "Architecture & Manufacturing:\n"
    "- Must use the Blackwell GPU architecture\n"
    "- Must be manufactured on TSMC's 4NP process node\n"
    "- Die size must be exactly 750 mm²\n"
    "- Transistor count must be exactly 92.2 billion\n\n"
    "Compute Core Configuration:\n"
    "- Must have exactly 21,760 CUDA cores\n"
    "- Must have exactly 170 Streaming Multiprocessors (SMs), with each SM containing 128 CUDA cores\n"
    "- Must have 680 fifth-generation Tensor Cores\n"
    "- The Tensor Cores must support FP4 precision operations\n\n"
    "Memory Subsystem:\n"
    "- Must have 32 GB of GDDR7 memory\n"
    "- Memory interface must be 512-bit wide\n"
    "- Memory bandwidth must be 1,792 GB/sec\n"
    "- Memory speed must be 28 Gbps GDDR7\n"
    "- Must use PAM3 (Pulse Amplitude Modulation 3-level) signaling technology\n\n"
    "Architectural Innovations:\n"
    "- Must include an AI Management Processor (AMP) implemented as a RISC-V processor for GPU context scheduling\n"
    "- Must support DLSS 4 with Multi Frame Generation capability (generating up to 3 additional frames per rendered frame)\n"
    "- Must have fourth-generation RT Cores with support for both Mega Geometry technology (with Cluster-level Acceleration Structures) "
    "and Linear Swept Spheres (LSS) for hair rendering\n\n"
    "Power & Connectivity:\n"
    "- Total Graphics Power (TGP) must be exactly 575 W\n"
    "- Must have 3x DisplayPort 2.1b outputs\n"
    "- Must have 1x HDMI 2.1b output\n"
    "- Must support PCI-Express 5.0 x16 interface\n\n"
    "Founders Edition physical constraints:\n"
    "- Dimensions must be exactly 304mm (L) × 137mm (H) × 40mm (W)\n"
    "- Must be dual-slot (exactly 2 expansion slots)\n\n"
    "What is the model name of the consumer graphics card that satisfies all these specifications? "
    "Provide the full official model designation and include reference URLs from credible sources (such as NVIDIA's official website, "
    "technical documentation, or reputable hardware review sites) that verify each category of specifications."
)

# Expected constraints (for clarity; used in claim strings)
EXPECTED = {
    "architecture": "Blackwell",
    "process_node": "TSMC 4NP",
    "die_size": "750 mm²",
    "transistor_count": "92.2 billion",
    "cuda_cores": "21,760",
    "sms": "170",
    "cuda_per_sm": "128",
    "tensor_cores": "680",
    "tensor_gen": "5th-generation",
    "fp4_support": "FP4",
    "memory_size": "32 GB",
    "memory_type": "GDDR7",
    "memory_interface": "512-bit",
    "memory_bandwidth": "1,792 GB/s",
    "memory_speed": "28 Gbps",
    "signaling": "PAM3",
    "amp": "AI Management Processor (AMP) implemented as a RISC-V processor for GPU context scheduling",
    "dlss4_mfg": "DLSS 4 with Multi Frame Generation up to 3 additional frames per rendered frame",
    "rt_core_gen": "Fourth-generation RT Cores",
    "rt_ray_tri_2x": "2× ray-triangle intersection throughput vs Ada",
    "rt_mega_geometry_clas": "Mega Geometry with Cluster-level Acceleration Structures (CLAS)",
    "rt_lss_hair": "Linear Swept Spheres (LSS) for hair rendering",
    "ser_2_0": "Shader Execution Reordering (SER) 2.0",
    "tgp": "575 W",
    "display_outputs": "3× DisplayPort 2.1b + 1× HDMI 2.1b",
    "pcie": "PCI-Express 5.0 x16",
    "fe_dimensions": "304 mm × 137 mm × 40 mm",
    "fe_slots": "2",
}


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class ArchMfgSpec(BaseModel):
    architecture: Optional[str] = None
    process_node: Optional[str] = None
    die_size: Optional[str] = None
    transistor_count: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ComputeSpec(BaseModel):
    cuda_cores: Optional[str] = None
    sms: Optional[str] = None
    cuda_per_sm: Optional[str] = None
    tensor_cores: Optional[str] = None
    tensor_core_generation: Optional[str] = None
    supports_fp4: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class MemorySpec(BaseModel):
    memory_size: Optional[str] = None
    memory_type: Optional[str] = None
    interface_width: Optional[str] = None
    bandwidth: Optional[str] = None
    memory_speed: Optional[str] = None
    signaling: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class InnovationsSpec(BaseModel):
    amp_riscv: Optional[str] = None
    dlss4_mfg: Optional[str] = None
    rt_core_generation: Optional[str] = None
    ray_triangle_2x_vs_ada: Optional[str] = None
    mega_geometry_clas: Optional[str] = None
    linear_swept_spheres: Optional[str] = None
    ser_2_0: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PowerConnectivitySpec(BaseModel):
    tgp: Optional[str] = None
    displayport_outputs: Optional[str] = None
    hdmi_output: Optional[str] = None
    pcie_interface: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class FoundersPhysicalSpec(BaseModel):
    dimensions_l_h_w_mm: Optional[str] = None
    slot_occupancy: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class GPUAnswerExtraction(BaseModel):
    model_name: Optional[str] = None
    model_urls: List[str] = Field(default_factory=list)

    arch_mfg: Optional[ArchMfgSpec] = None
    compute: Optional[ComputeSpec] = None
    memory: Optional[MemorySpec] = None
    innovations: Optional[InnovationsSpec] = None
    power_connectivity: Optional[PowerConnectivitySpec] = None
    founders_physical: Optional[FoundersPhysicalSpec] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_gpu_specs() -> str:
    return """
Extract the graphics card model and all category-specific specifications exactly as stated in the answer, along with the reference URLs that the answer provides to support each category. Return JSON fields as strings exactly as they appear in the answer text. If a field is not mentioned, return null for that field. For URLs, return only actual URLs that appear in the answer; do not invent any.

Fields to extract:
- model_name: Full official model designation of the graphics card claimed in the answer (e.g., "NVIDIA GeForce RTX 5090 Founders Edition"). If multiple appear, pick the main claimed model for the constraints.
- model_urls: An array of any URLs in the answer that identify the model (e.g., official product page or overview pages).

For each category below, extract both the specs (as strings) and the list of URLs used as references for that category (the URLs that the answer associates with that category):

1) arch_mfg:
   - architecture
   - process_node
   - die_size
   - transistor_count
   - urls (list)

2) compute:
   - cuda_cores
   - sms
   - cuda_per_sm
   - tensor_cores
   - tensor_core_generation
   - supports_fp4
   - urls (list)

3) memory:
   - memory_size
   - memory_type
   - interface_width
   - bandwidth
   - memory_speed
   - signaling
   - urls (list)

4) innovations:
   - amp_riscv
   - dlss4_mfg
   - rt_core_generation
   - ray_triangle_2x_vs_ada
   - mega_geometry_clas
   - linear_swept_spheres
   - ser_2_0
   - urls (list)

5) power_connectivity:
   - tgp
   - displayport_outputs
   - hdmi_output
   - pcie_interface
   - urls (list)

6) founders_physical:
   - dimensions_l_h_w_mm
   - slot_occupancy
   - urls (list)

Rules:
- Return strings as they appear in the answer (e.g., keep units and punctuation like "750 mm²", "92.2B", "512-bit", "1,792 GB/s", "28 Gbps", etc.).
- For booleans or capability statements (e.g., FP4 support), still return strings like "supports FP4" or "FP4" if present; otherwise null.
- For URL arrays, include only valid, explicit URLs mentioned in the answer (plain, markdown, or embedded).
- If the answer lists multiple URLs per category, include them all in that category's urls array.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def _aggregate_all_urls(ex: GPUAnswerExtraction) -> List[str]:
    urls: List[str] = []
    urls.extend(_safe_urls(ex.model_urls))
    if ex.arch_mfg:
        urls.extend(_safe_urls(ex.arch_mfg.urls))
    if ex.compute:
        urls.extend(_safe_urls(ex.compute.urls))
    if ex.memory:
        urls.extend(_safe_urls(ex.memory.urls))
    if ex.innovations:
        urls.extend(_safe_urls(ex.innovations.urls))
    if ex.power_connectivity:
        urls.extend(_safe_urls(ex.power_connectivity.urls))
    if ex.founders_physical:
        urls.extend(_safe_urls(ex.founders_physical.urls))
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


async def _verify_with_leaf(
    evaluator: Evaluator,
    *,
    leaf_id: str,
    desc: str,
    parent,
    claim: str,
    sources: Optional[List[str]],
    additional_instruction: str,
    extra_prerequisites: Optional[List[Any]] = None,
    critical: bool = True,
):
    node = evaluator.add_leaf(
        id=leaf_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources if sources else None,
        additional_instruction=additional_instruction,
        extra_prerequisites=extra_prerequisites,
    )
    return node


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_model_identification(
    evaluator: Evaluator,
    parent,
    ex: GPUAnswerExtraction
):
    node = evaluator.add_parallel(
        id="model_identification",
        desc="Provide the full official model designation of the consumer graphics card being claimed.",
        parent=parent,
        critical=True,
    )

    # Leaf 1: Official model designation provided
    model_present = bool(ex.model_name and ex.model_name.strip())
    evaluator.add_custom_node(
        result=model_present,
        id="official_model_designation_provided",
        desc="Answer includes the full official model designation (model name) of the graphics card.",
        parent=node,
        critical=True
    )

    # Leaf 2: Is consumer graphics card
    all_urls = _aggregate_all_urls(ex)
    model_for_claim = ex.model_name or "the claimed graphics card model in the answer"
    await _verify_with_leaf(
        evaluator,
        leaf_id="is_consumer_graphics_card",
        desc="The identified model is a consumer graphics card (not a data-center-only product).",
        parent=node,
        claim=f"{model_for_claim} is a consumer desktop graphics card intended for end-users (e.g., GeForce/TITAN/consumer RTX class), not a data-center-only product.",
        sources=all_urls,
        additional_instruction=(
            "Verify that the product belongs to a consumer/desktop GPU product line (e.g., NVIDIA GeForce, Founders Edition, or AIB cards) "
            "rather than a data center product (e.g., NVIDIA Tesla/Hopper/Blackwell B-series for data centers, RTX A-series workstation). "
            "Accept official product pages or reputable hardware sites (e.g., nvidia.com, techpowerup.com, tomshardware.com, anandtech.com, "
            "pcgamer.com, videocards.com, guru3d.com, etc.) as credible sources."
        ),
        extra_prerequisites=None
    )
    return node


async def build_arch_mfg(
    evaluator: Evaluator,
    parent,
    model_name: str,
    ex: GPUAnswerExtraction
):
    cat = evaluator.add_parallel(
        id="architecture_and_manufacturing",
        desc="Architecture & Manufacturing constraints are satisfied, with credible references.",
        parent=parent,
        critical=True
    )

    urls = _safe_urls(ex.arch_mfg.urls if ex.arch_mfg else [])

    # Refs presence check (critical gate for this category)
    refs_node = evaluator.add_custom_node(
        result=len(urls) > 0,
        id="refs_architecture_and_manufacturing",
        desc="Provides credible reference URL(s) that collectively verify the Architecture & Manufacturing specs above.",
        parent=cat,
        critical=True
    )

    # Specs leaves (each depends on refs_node)
    await _verify_with_leaf(
        evaluator,
        leaf_id="blackwell_architecture",
        desc="GPU uses the Blackwell architecture.",
        parent=cat,
        claim=f"The {model_name} uses NVIDIA Blackwell architecture.",
        sources=urls,
        additional_instruction="Accept mentions like 'NVIDIA Blackwell architecture' or 'Blackwell-based GPU'.",
        extra_prerequisites=[refs_node]
    )
    await _verify_with_leaf(
        evaluator,
        leaf_id="tsmc_4np_process",
        desc="GPU is manufactured on TSMC's 4NP process node.",
        parent=cat,
        claim=f"The {model_name} is manufactured on TSMC 4NP process node.",
        sources=urls,
        additional_instruction="Allow minor formatting variants like 'TSMC 4N P' or 'TSMC 4N performance-optimized'. It must clearly indicate '4NP'.",
        extra_prerequisites=[refs_node]
    )
    await _verify_with_leaf(
        evaluator,
        leaf_id="die_size_750_mm2",
        desc="Die size is exactly 750 mm².",
        parent=cat,
        claim=f"The {model_name} has a die size of 750 mm².",
        sources=urls,
        additional_instruction="Treat 'mm2' and 'mm²' as equivalent.",
        extra_prerequisites=[refs_node]
    )
    await _verify_with_leaf(
        evaluator,
        leaf_id="transistor_count_92_2_billion",
        desc="Transistor count is exactly 92.2 billion.",
        parent=cat,
        claim=f"The {model_name} has 92.2 billion transistors (i.e., ~92.2B).",
        sources=urls,
        additional_instruction="Allow reasonable numeric formatting (e.g., '92.2B' == '92.2 billion').",
        extra_prerequisites=[refs_node]
    )
    return cat


async def build_compute(
    evaluator: Evaluator,
    parent,
    model_name: str,
    ex: GPUAnswerExtraction
):
    cat = evaluator.add_parallel(
        id="compute_core_configuration",
        desc="Compute Core Configuration constraints are satisfied, with credible references.",
        parent=parent,
        critical=True
    )
    urls = _safe_urls(ex.compute.urls if ex.compute else [])

    refs_node = evaluator.add_custom_node(
        result=len(urls) > 0,
        id="refs_compute_core_configuration",
        desc="Provides credible reference URL(s) that collectively verify the Compute Core Configuration specs above.",
        parent=cat,
        critical=True
    )

    await _verify_with_leaf(
        evaluator,
        leaf_id="cuda_cores_21760",
        desc="Has exactly 21,760 CUDA cores.",
        parent=cat,
        claim=f"The {model_name} has exactly 21,760 CUDA cores.",
        sources=urls,
        additional_instruction="Allow thousand separators or plain digits (e.g., '21,760' vs '21760').",
        extra_prerequisites=[refs_node]
    )
    await _verify_with_leaf(
        evaluator,
        leaf_id="sms_170",
        desc="Has exactly 170 Streaming Multiprocessors (SMs).",
        parent=cat,
        claim=f"The {model_name} has exactly 170 Streaming Multiprocessors (SMs).",
        sources=urls,
        additional_instruction="Variants like 'SMs'/'SM' are acceptable; value must be 170.",
        extra_prerequisites=[refs_node]
    )
    await _verify_with_leaf(
        evaluator,
        leaf_id="cuda_cores_per_sm_128",
        desc="Each SM contains exactly 128 CUDA cores.",
        parent=cat,
        claim=f"Each SM on the {model_name} contains 128 CUDA cores.",
        sources=urls,
        additional_instruction="Explicit per-SM CUDA core count should be 128.",
        extra_prerequisites=[refs_node]
    )
    await _verify_with_leaf(
        evaluator,
        leaf_id="tensor_cores_680_5th_gen",
        desc="Has exactly 680 fifth-generation Tensor Cores.",
        parent=cat,
        claim=f"The {model_name} has 680 fifth-generation Tensor Cores.",
        sources=urls,
        additional_instruction="Must mention both the count (680) and 5th generation association, directly or clearly implied for this model.",
        extra_prerequisites=[refs_node]
    )
    await _verify_with_leaf(
        evaluator,
        leaf_id="tensor_fp4_support",
        desc="Tensor Cores support FP4 precision operations.",
        parent=cat,
        claim=f"The Tensor Cores on the {model_name} support FP4 precision operations.",
        sources=urls,
        additional_instruction="Accept mentions like 'FP4', '4-bit floating point', or equivalent standardized naming.",
        extra_prerequisites=[refs_node]
    )
    return cat


async def build_memory(
    evaluator: Evaluator,
    parent,
    model_name: str,
    ex: GPUAnswerExtraction
):
    cat = evaluator.add_parallel(
        id="memory_subsystem",
        desc="Memory Subsystem constraints are satisfied, with credible references.",
        parent=parent,
        critical=True
    )
    urls = _safe_urls(ex.memory.urls if ex.memory else [])

    refs_node = evaluator.add_custom_node(
        result=len(urls) > 0,
        id="refs_memory_subsystem",
        desc="Provides credible reference URL(s) that collectively verify the Memory Subsystem specs above.",
        parent=cat,
        critical=True
    )

    await _verify_with_leaf(
        evaluator,
        leaf_id="memory_32gb_gddr7",
        desc="Has 32 GB of GDDR7 memory.",
        parent=cat,
        claim=f"The {model_name} is equipped with 32 GB of GDDR7 memory.",
        sources=urls,
        additional_instruction="Must explicitly indicate 32 GB and GDDR7.",
        extra_prerequisites=[refs_node]
    )
    await _verify_with_leaf(
        evaluator,
        leaf_id="memory_interface_512_bit",
        desc="Memory interface is 512-bit wide.",
        parent=cat,
        claim=f"The {model_name} has a 512-bit memory interface.",
        sources=urls,
        additional_instruction="Accept '512-bit' with or without hyphen; the width must be 512.",
        extra_prerequisites=[refs_node]
    )
    await _verify_with_leaf(
        evaluator,
        leaf_id="memory_bandwidth_1792_gb_per_sec",
        desc="Memory bandwidth is 1,792 GB/sec.",
        parent=cat,
        claim=f"The {model_name} provides memory bandwidth of 1,792 GB/s.",
        sources=urls,
        additional_instruction="Allow 'GB/s' vs 'GB/sec' and comma separators.",
        extra_prerequisites=[refs_node]
    )
    await _verify_with_leaf(
        evaluator,
        leaf_id="memory_speed_28_gbps",
        desc="Memory speed is 28 Gbps GDDR7.",
        parent=cat,
        claim=f"The GDDR7 memory on the {model_name} runs at 28 Gbps.",
        sources=urls,
        additional_instruction="Exact value must be 28 Gbps; allow 'Gbps' capitalization variance.",
        extra_prerequisites=[refs_node]
    )
    await _verify_with_leaf(
        evaluator,
        leaf_id="pam3_signaling",
        desc="Uses PAM3 (Pulse Amplitude Modulation 3-level) signaling technology.",
        parent=cat,
        claim=f"The {model_name} uses PAM3 (3‑level Pulse Amplitude Modulation) signaling for its memory interface.",
        sources=urls,
        additional_instruction="Accept explicit 'PAM3' or clearly equivalent phrasing (3-level PAM).",
        extra_prerequisites=[refs_node]
    )
    return cat


async def build_innovations(
    evaluator: Evaluator,
    parent,
    model_name: str,
    ex: GPUAnswerExtraction
):
    cat = evaluator.add_parallel(
        id="architectural_innovations",
        desc="Architectural Innovations constraints are satisfied, with credible references.",
        parent=parent,
        critical=True
    )
    urls = _safe_urls(ex.innovations.urls if ex.innovations else [])

    refs_node = evaluator.add_custom_node(
        result=len(urls) > 0,
        id="refs_architectural_innovations",
        desc="Provides credible reference URL(s) that collectively verify the Architectural Innovations specs above.",
        parent=cat,
        critical=True
    )

    await _verify_with_leaf(
        evaluator,
        leaf_id="amp_riscv",
        desc="Includes an AI Management Processor (AMP) implemented as a RISC-V processor for GPU context scheduling.",
        parent=cat,
        claim=f"The {model_name} includes an AI Management Processor (AMP) implemented as a RISC-V processor for GPU context scheduling.",
        sources=urls,
        additional_instruction="Look for 'AMP' and 'RISC-V' together related to GPU scheduler/context management on this model.",
        extra_prerequisites=[refs_node]
    )
    await _verify_with_leaf(
        evaluator,
        leaf_id="dlss4_multi_frame_generation",
        desc="Supports DLSS 4 with Multi Frame Generation capability (up to 3 additional frames per rendered frame).",
        parent=cat,
        claim=f"The {model_name} supports DLSS 4 with Multi Frame Generation that can generate up to 3 additional frames per rendered frame.",
        sources=urls,
        additional_instruction="Must explicitly indicate DLSS 4 and multi-frame generation up to +3 frames.",
        extra_prerequisites=[refs_node]
    )
    await _verify_with_leaf(
        evaluator,
        leaf_id="rt_cores_4th_gen",
        desc="Has fourth-generation RT Cores.",
        parent=cat,
        claim=f"The {model_name} has fourth-generation RT Cores.",
        sources=urls,
        additional_instruction="Accept '4th-gen RT Cores' or 'fourth generation ray tracing cores'.",
        extra_prerequisites=[refs_node]
    )
    await _verify_with_leaf(
        evaluator,
        leaf_id="rt_ray_triangle_2x_vs_ada",
        desc="Fourth-generation RT Cores have 2× ray-triangle intersection throughput compared to Ada architecture.",
        parent=cat,
        claim=f"The fourth-generation RT Cores on the {model_name} provide 2× ray‑triangle intersection throughput compared to Ada architecture.",
        sources=urls,
        additional_instruction="Must explicitly indicate a 2× improvement vs Ada for ray-triangle intersection throughput.",
        extra_prerequisites=[refs_node]
    )
    await _verify_with_leaf(
        evaluator,
        leaf_id="rt_mega_geometry_clas",
        desc="RT Cores support Mega Geometry technology with Cluster-level Acceleration Structures (CLAS).",
        parent=cat,
        claim=f"The RT Cores on the {model_name} support Mega Geometry with Cluster‑level Acceleration Structures (CLAS).",
        sources=urls,
        additional_instruction="Look for 'Mega Geometry' and 'CLAS' support explicitly.",
        extra_prerequisites=[refs_node]
    )
    await _verify_with_leaf(
        evaluator,
        leaf_id="rt_linear_swept_spheres_lss",
        desc="RT Cores support Linear Swept Spheres (LSS) for hair rendering.",
        parent=cat,
        claim=f"The RT Cores on the {model_name} support Linear Swept Spheres (LSS) for hair rendering.",
        sources=urls,
        additional_instruction="Must mention Linear Swept Spheres or LSS for hair rendering on RT hardware.",
        extra_prerequisites=[refs_node]
    )
    await _verify_with_leaf(
        evaluator,
        leaf_id="ser_2_0",
        desc="Supports Shader Execution Reordering (SER) 2.0.",
        parent=cat,
        claim=f"The {model_name} supports Shader Execution Reordering (SER) 2.0.",
        sources=urls,
        additional_instruction="Accept 'SER 2.0' or 'Shader Execution Reordering 2.0'.",
        extra_prerequisites=[refs_node]
    )
    return cat


async def build_power_connectivity(
    evaluator: Evaluator,
    parent,
    model_name: str,
    ex: GPUAnswerExtraction
):
    cat = evaluator.add_parallel(
        id="power_and_connectivity",
        desc="Power & Connectivity constraints are satisfied, with credible references.",
        parent=parent,
        critical=True
    )
    urls = _safe_urls(ex.power_connectivity.urls if ex.power_connectivity else [])

    refs_node = evaluator.add_custom_node(
        result=len(urls) > 0,
        id="refs_power_and_connectivity",
        desc="Provides credible reference URL(s) that collectively verify the Power & Connectivity specs above.",
        parent=cat,
        critical=True
    )

    await _verify_with_leaf(
        evaluator,
        leaf_id="tgp_575_w",
        desc="Total Graphics Power (TGP) is exactly 575 W.",
        parent=cat,
        claim=f"The {model_name} has a Total Graphics Power (TGP) of 575 W.",
        sources=urls,
        additional_instruction="Accept 'TGP' wording; value must be 575 W.",
        extra_prerequisites=[refs_node]
    )
    await _verify_with_leaf(
        evaluator,
        leaf_id="display_outputs",
        desc="Has 3× DisplayPort 2.1b outputs and 1× HDMI 2.1b output.",
        parent=cat,
        claim=f"The {model_name} provides 3 DisplayPort 2.1b outputs and 1 HDMI 2.1b output.",
        sources=urls,
        additional_instruction="Allow minor naming variations like 'DP 2.1' vs 'DP 2.1b' and 'HDMI 2.1' vs 'HDMI 2.1b' if otherwise credible and clearly the latest revision.",
        extra_prerequisites=[refs_node]
    )
    await _verify_with_leaf(
        evaluator,
        leaf_id="pcie_5_0_x16",
        desc="Supports PCI-Express 5.0 ×16 interface.",
        parent=cat,
        claim=f"The {model_name} supports PCIe 5.0 x16 interface.",
        sources=urls,
        additional_instruction="Accept 'PCIe 5.0 x16' vs 'PCI‑Express 5.0 x16' naming variants.",
        extra_prerequisites=[refs_node]
    )
    return cat


async def build_founders_physical(
    evaluator: Evaluator,
    parent,
    model_name: str,
    ex: GPUAnswerExtraction
):
    cat = evaluator.add_parallel(
        id="founders_edition_physical_constraints",
        desc="Founders Edition physical constraints from the constraints list are satisfied, with credible references.",
        parent=parent,
        critical=True
    )
    urls = _safe_urls(ex.founders_physical.urls if ex.founders_physical else [])

    refs_node = evaluator.add_custom_node(
        result=len(urls) > 0,
        id="refs_founders_edition_physical",
        desc="Provides credible reference URL(s) that collectively verify the Founders Edition physical specs above.",
        parent=cat,
        critical=True
    )

    await _verify_with_leaf(
        evaluator,
        leaf_id="fe_dimensions",
        desc="Founders Edition dimensions are exactly 304mm length × 137mm height × 40mm width.",
        parent=cat,
        claim=f"The Founders Edition of the {model_name} measures 304 mm (L) × 137 mm (H) × 40 mm (W).",
        sources=urls,
        additional_instruction="Treat 'mm' units strictly; minor punctuation or spacing variants are acceptable.",
        extra_prerequisites=[refs_node]
    )
    await _verify_with_leaf(
        evaluator,
        leaf_id="fe_dual_slot",
        desc="Founders Edition occupies exactly 2 expansion slots (dual-slot design).",
        parent=cat,
        claim=f"The Founders Edition of the {model_name} is a dual-slot card (occupies exactly 2 expansion slots).",
        sources=urls,
        additional_instruction="Accept 'dual-slot' or '2-slot' as equivalently indicating exactly 2 slots.",
        extra_prerequisites=[refs_node]
    )
    return cat


async def build_specs_and_references(
    evaluator: Evaluator,
    parent,
    ex: GPUAnswerExtraction
):
    node = evaluator.add_parallel(
        id="specs_and_references",
        desc="Verify the identified model satisfies every constraint, with credible references per category.",
        parent=parent,
        critical=True
    )
    model_name = ex.model_name or "the claimed graphics card model"

    # Build each category subtree
    await build_arch_mfg(evaluator, node, model_name, ex)
    await build_compute(evaluator, node, model_name, ex)
    await build_memory(evaluator, node, model_name, ex)
    await build_innovations(evaluator, node, model_name, ex)
    await build_power_connectivity(evaluator, node, model_name, ex)
    await build_founders_physical(evaluator, node, model_name, ex)

    return node


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # First identify model, then verify specs
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

    # Record the expected constraints for transparency
    evaluator.add_custom_info(
        info={"expected_constraints": EXPECTED},
        info_type="constraints",
        info_name="expected_constraints"
    )

    # 1) Extract structured data from the answer
    extraction: GPUAnswerExtraction = await evaluator.extract(
        prompt=prompt_extract_gpu_specs(),
        template_class=GPUAnswerExtraction,
        extraction_name="gpu_spec_extraction"
    )

    # 2) Build verification tree
    # Root is critical & sequential: if identification fails, all spec checks will be skipped automatically.
    # 2.1 Model identification
    await build_model_identification(evaluator, root, extraction)

    # 2.2 Specs and references per category
    await build_specs_and_references(evaluator, root, extraction)

    # 3) Return summary
    return evaluator.get_summary()