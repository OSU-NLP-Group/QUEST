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
TASK_ID = "datacenter_ai_chip_2024_specs_check"
TASK_DESCRIPTION = """As of December 2024, identify the name and full model designation of the datacenter AI accelerator chip that meets ALL of the following specifications:

1. The chip was released or officially announced in 2024
2. It delivers between 2,500 and 3,000 TOPS (trillion operations per second) for INT8 operations, measured without sparsity acceleration techniques
3. It uses HBM3E (High Bandwidth Memory 3 Enhanced) memory technology
4. It provides memory bandwidth of at least 5,500 GB/s
5. It has a total memory capacity of at least 150GB
6. Its compute die is manufactured using TSMC's N5 (5nm) process node
7. It has a TDP (Thermal Design Power) rating of exactly 750 watts
8. It belongs to a product family where the immediately preceding generation model used HBM3 memory (not HBM3E)
9. It is manufactured by one of the top 3 datacenter GPU/accelerator manufacturers by market valuation as of 2024
10. It is explicitly designed and marketed for datacenter AI workloads, supporting both training and inference
11. It supports hardware-accelerated structured sparsity with a 2:4 sparsity pattern
12. It supports multiple precision formats including FP8, FP16, BF16, and INT8 data types

Provide the chip's complete commercial model name/number and a reference URL from the manufacturer or a credible technology publication that confirms these specifications.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ChipSpecExtraction(BaseModel):
    """Structured information extracted from the agent's answer."""
    model_name: Optional[str] = None
    vendor: Optional[str] = None
    family_name: Optional[str] = None

    release_info: Optional[str] = None  # e.g., "Announced June 2024"
    int8_tops: Optional[str] = None     # as stated; may include units or qualifiers
    int8_tops_context: Optional[str] = None  # e.g., "without sparsity", "with 2:4 sparsity"

    memory_type: Optional[str] = None       # e.g., "HBM3E"
    memory_bandwidth: Optional[str] = None  # e.g., "6 TB/s", "6000 GB/s"
    memory_capacity: Optional[str] = None   # e.g., "192GB"

    process_node: Optional[str] = None      # e.g., "TSMC N5", "5nm"
    tdp_watts: Optional[str] = None         # e.g., "750W"

    previous_gen_model: Optional[str] = None
    previous_gen_memory: Optional[str] = None  # e.g., "HBM3"

    datacenter_ai_segment_desc: Optional[str] = None  # wording indicating datacenter AI, training + inference
    training_inference_support: Optional[str] = None  # explicit mention of both training and inference

    sparsity_2_4_desc: Optional[str] = None  # wording that indicates 2:4 structured sparsity support

    precision_formats: List[str] = Field(default_factory=list)  # e.g., ["FP8", "FP16", "BF16", "INT8"]

    reference_urls: List[str] = Field(default_factory=list)     # manufacturer or credible tech publication URLs


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_chip_spec() -> str:
    return """
    Extract the chip identification and all relevant specification details from the answer. We need the exact model designation and any provided specifications and sources.

    Return a JSON object with the following fields:
    - model_name: The chip’s full commercial model name/number as written in the answer (e.g., "Instinct MI325X", "NVIDIA B200", etc.). If absent, return null.
    - vendor: The manufacturer’s name (e.g., "AMD", "NVIDIA", "Intel"). If absent, return null.
    - family_name: The product family or series this chip belongs to (e.g., "Instinct", "Blackwell"). If absent, return null.

    - release_info: The release or official announcement timing (e.g., "announced in June 2024", "released Q4 2024") exactly as described in the answer. If absent, return null.
    - int8_tops: The INT8 throughput figure as quoted (e.g., "2600 TOPS"), if provided. If absent, return null.
    - int8_tops_context: Any qualifier indicating measurement conditions (e.g., "without sparsity", "with structured sparsity", "2:4 sparsity"). If absent, return null.

    - memory_type: The memory technology used (e.g., "HBM3E", "HBM3"). If absent, return null.
    - memory_bandwidth: The memory bandwidth figure as written (e.g., "6 TB/s", "6000 GB/s"). If absent, return null.
    - memory_capacity: The total memory capacity as written (e.g., "192GB"). If absent, return null.

    - process_node: The manufacturing process node of the compute die (e.g., "TSMC N5", "5nm"). If absent, return null.
    - tdp_watts: TDP value as written (e.g., "750W"). If absent, return null.

    - previous_gen_model: The immediately preceding generation model name/number, if provided (e.g., "MI300X", "H100"). If absent, return null.
    - previous_gen_memory: The memory technology of the preceding model (e.g., "HBM3"). If absent, return null.

    - datacenter_ai_segment_desc: Any statement that the chip targets datacenter AI workloads (e.g., "datacenter AI accelerator"). If absent, return null.
    - training_inference_support: Any explicit statement the chip supports both training and inference. If absent, return null.

    - sparsity_2_4_desc: Any statement indicating hardware-accelerated structured sparsity with a 2:4 pattern. If absent, return null.

    - precision_formats: A list of precision formats mentioned (e.g., ["FP8", "FP16", "BF16", "INT8"]). If none mentioned, return an empty list.

    - reference_urls: Extract all URLs cited as references in the answer that relate to this chip/specifications. These can be manufacturer pages (e.g., amd.com, nvidia.com, intel.com) or credible publications (e.g., anandtech.com, tomshardware.com, semianalysis.com, phoronix.com). Return only actual URLs; do not include plain text mentions. If no URLs are provided, return an empty list.

    Notes:
    - Extract exactly what appears in the answer. Do not infer or invent.
    - For URL extraction, follow the SPECIAL RULES FOR URL SOURCES EXTRACTION and SPECIAL RULES FOR URL EXTRACTION.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _chip_label(extracted: ChipSpecExtraction) -> str:
    if extracted.model_name and extracted.model_name.strip():
        return extracted.model_name.strip()
    return "the identified datacenter AI accelerator chip"

def _vendor_label(extracted: ChipSpecExtraction) -> str:
    return (extracted.vendor or "").strip()


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: ChipSpecExtraction, root_node) -> None:
    """
    Build the verification tree based on the rubric and run verifications.
    The root node is critical and aggregates children in parallel.
    """

    # 1) Existence / identification check (critical)
    exists_node = evaluator.add_custom_node(
        result=bool(extracted.model_name and extracted.model_name.strip()) and bool(extracted.reference_urls),
        id="Chip_Model_Designation_Provided",
        desc="Answer provides the chip's complete commercial model name/number (full model designation) and at least one reference URL",
        parent=root_node,
        critical=True
    )

    # 2) Reference documentation credibility (critical) – must precede other verifications
    ref_doc_node = evaluator.add_leaf(
        id="Reference_Documentation",
        desc="At least one referenced URL is an official manufacturer page or a credible technology publication that supports the specifications",
        parent=root_node,
        critical=True
    )
    # Verify credibility via multi-URL check; pass if any URL is credible and relevant
    await evaluator.verify(
        claim="This page is either an official manufacturer source or a credible technology publication that discusses this chip and its specifications.",
        node=ref_doc_node,
        sources=extracted.reference_urls,
        additional_instruction="Treat domains like amd.com, nvidia.com, intel.com as manufacturer; treat well-known tech publications (e.g., anandtech.com, tomshardware.com, semianalysis.com, phoronix.com, wccftech.com) as credible. The page should discuss the identified chip and its specs."
    )

    # Prepare remaining checks; their preconditions will automatically depend on critical siblings
    chip_name = _chip_label(extracted)
    vendor_name = _vendor_label(extracted)

    # 3) Release year 2024
    release_node = evaluator.add_leaf(
        id="Release_Year_2024",
        desc="The chip was released or officially announced in 2024",
        parent=root_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The chip {chip_name} was released or officially announced in calendar year 2024.",
        node=release_node,
        sources=extracted.reference_urls,
        additional_instruction="Accept product announcement, launch press release, or credible reports stating the official announcement in 2024. Check page dates or explicit wording."
    )

    # 4) INT8 performance range 2,500–3,000 TOPS (without sparsity)
    perf_node = evaluator.add_leaf(
        id="Performance_Range",
        desc="The chip delivers between 2,500 and 3,000 TOPS for INT8 operations, measured without sparsity acceleration",
        parent=root_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The chip {chip_name} achieves between 2,500 and 3,000 INT8 TOPS without using any sparsity acceleration (i.e., dense).",
        node=perf_node,
        sources=extracted.reference_urls,
        additional_instruction="Verify the stated INT8 throughput refers to dense performance (no 2:4 sparsity). If the page only lists values 'with sparsity', this claim should be considered not supported."
    )

    # 5) HBM3E memory technology
    hbm3e_node = evaluator.add_leaf(
        id="HBM3E_Memory",
        desc="The chip uses HBM3E (High Bandwidth Memory 3 Enhanced) memory technology",
        parent=root_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The chip {chip_name} uses HBM3E memory technology.",
        node=hbm3e_node,
        sources=extracted.reference_urls,
        additional_instruction="Look for 'HBM3E' explicitly in specifications."
    )

    # 6) Memory bandwidth ≥ 5,500 GB/s
    bandwidth_node = evaluator.add_leaf(
        id="Memory_Bandwidth_Minimum",
        desc="The chip provides memory bandwidth of at least 5,500 GB/s",
        parent=root_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The chip {chip_name} provides memory bandwidth of at least 5,500 GB/s.",
        node=bandwidth_node,
        sources=extracted.reference_urls,
        additional_instruction="If bandwidth is given in TB/s, convert: 1 TB/s = 1000 GB/s. Pass if the figure is ≥ 5,500 GB/s."
    )

    # 7) Memory capacity ≥ 150 GB
    capacity_node = evaluator.add_leaf(
        id="Memory_Capacity_Minimum",
        desc="The chip has total memory capacity of at least 150GB",
        parent=root_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The chip {chip_name} has total memory capacity of at least 150 GB.",
        node=capacity_node,
        sources=extracted.reference_urls,
        additional_instruction="If capacity is stated in GB, compare directly; pass if ≥ 150 GB."
    )

    # 8) TSMC N5 (5nm) process node
    process_node_leaf = evaluator.add_leaf(
        id="TSMC_N5_Process",
        desc="The compute die is manufactured using TSMC's N5 (5nm) process node",
        parent=root_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The compute die for {chip_name} is manufactured using TSMC's N5 (5nm) process.",
        node=process_node_leaf,
        sources=extracted.reference_urls,
        additional_instruction="Accept wording 'TSMC N5', '5nm' specifically tied to the compute die. Do not accept 'N4' or '4N' as equivalent."
    )

    # 9) TDP exactly 750W
    tdp_node = evaluator.add_leaf(
        id="Power_TDP_750W",
        desc="The chip's TDP (Thermal Design Power) is exactly 750 watts",
        parent=root_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The chip {chip_name} has a TDP of exactly 750 watts.",
        node=tdp_node,
        sources=extracted.reference_urls,
        additional_instruction="Pass only if the TDP figure is exactly 750W (not ~750W or a range); minor formatting variants (e.g., '750 W') are fine."
    )

    # 10) Product family evolution: previous gen used HBM3 (not HBM3E)
    family_node = evaluator.add_leaf(
        id="Product_Family_Evolution",
        desc="The chip belongs to a product family where the immediately preceding generation model used HBM3 (not HBM3E) memory",
        parent=root_node,
        critical=True
    )
    prev_label = extracted.previous_gen_model or "the immediately preceding generation model"
    await evaluator.verify(
        claim=f"In the same product family as {chip_name}, {prev_label} used HBM3 memory (not HBM3E).",
        node=family_node,
        sources=extracted.reference_urls,
        additional_instruction="Confirm the memory type of the directly preceding generation is HBM3 (not HBM3E). The page(s) should explicitly mention the preceding product and its memory."
    )

    # 11) Top three vendor (NVIDIA, AMD, Intel)
    vendor_top3_node = evaluator.add_leaf(
        id="Top_Three_Vendor",
        desc="The chip is manufactured by one of the top 3 datacenter GPU/accelerator manufacturers by market valuation as of 2024 (NVIDIA, AMD, or Intel)",
        parent=root_node,
        critical=True
    )
    # If vendor name extracted, use simple verification; otherwise rely on sources and wording
    if vendor_name:
        await evaluator.verify(
            claim=f"The manufacturer '{vendor_name}' is one of NVIDIA, AMD, or Intel.",
            node=vendor_top3_node,
            additional_instruction="Ignore case and minor variants. Pass if vendor_name ∈ {NVIDIA, AMD, Intel}."
        )
    else:
        await evaluator.verify(
            claim=f"The chip {chip_name} is manufactured by one of NVIDIA, AMD, or Intel.",
            node=vendor_top3_node,
            sources=extracted.reference_urls,
            additional_instruction="Determine from the page whether the manufacturer is NVIDIA, AMD, or Intel."
        )

    # 12) Datacenter AI segment + supports both training and inference
    segment_node = evaluator.add_leaf(
        id="Datacenter_AI_Segment",
        desc="The chip is explicitly designed and marketed for datacenter AI workloads, supporting both training and inference",
        parent=root_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The chip {chip_name} is explicitly marketed for datacenter AI workloads and supports both training and inference.",
        node=segment_node,
        sources=extracted.reference_urls,
        additional_instruction="Look for explicit wording indicating datacenter AI use and support for both training and inference."
    )

    # 13) 2:4 structured sparsity support
    sparsity_node = evaluator.add_leaf(
        id="Sparsity_Support",
        desc="The chip supports hardware-accelerated structured sparsity with a 2:4 sparsity pattern",
        parent=root_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The chip {chip_name} supports hardware-accelerated 2:4 structured sparsity.",
        node=sparsity_node,
        sources=extracted.reference_urls,
        additional_instruction="Look for '2:4 sparsity' or equivalent phrasing (e.g., 'two-out-of-four structured sparsity') and that it is hardware-accelerated."
    )

    # 14) Multi-precision support: FP8, FP16, BF16, INT8
    precision_node = evaluator.add_leaf(
        id="Multi_Precision_Support",
        desc="The chip supports multiple precision formats including FP8, FP16, BF16, and INT8 data types",
        parent=root_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The chip {chip_name} supports FP8, FP16, BF16, and INT8 data types.",
        node=precision_node,
        sources=extracted.reference_urls,
        additional_instruction="Pass only if all four formats (FP8, FP16, BF16, INT8) are supported."
    )


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
    Evaluate the agent's answer against the 2024 datacenter AI accelerator chip specifications.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates multiple critical checks independently
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

    # Extract structured chip information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_chip_spec(),
        template_class=ChipSpecExtraction,
        extraction_name="chip_spec_extraction",
    )

    # Record accepted top-3 vendors as ground truth context
    evaluator.add_ground_truth({
        "accepted_top3_vendors": ["NVIDIA", "AMD", "Intel"],
        "requirement_summary": [
            "Release/announcement in 2024",
            "INT8 TOPS between 2,500 and 3,000 (without sparsity)",
            "HBM3E memory",
            "Memory bandwidth ≥ 5,500 GB/s",
            "Memory capacity ≥ 150 GB",
            "TSMC N5 (5nm) process for compute die",
            "TDP exactly 750 W",
            "Previous generation used HBM3 (not HBM3E)",
            "Manufacturer is one of NVIDIA/AMD/Intel",
            "Datacenter AI workloads; supports both training and inference",
            "2:4 structured sparsity (hardware-accelerated)",
            "Supports FP8, FP16, BF16, INT8"
        ]
    })

    # Build and run verifications
    target_node = evaluator.add_parallel(
        id="Target_AI_Chip",
        desc="Identify the datacenter AI chip and provide required identifying information that satisfies all specified constraints",
        parent=root,
        critical=True
    )
    await build_and_verify_tree(evaluator, extracted, target_node)

    # Return the structured summary
    return evaluator.get_summary()