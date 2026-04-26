import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gtc_2026_nextgen_platform_and_specs"
TASK_DESCRIPTION = """
NVIDIA held its annual GTC conference from March 16-19, 2026 in San Jose, where the company unveiled its next-generation AI computing platform designed to succeed the Blackwell architecture. Identify the name of this new platform and provide the following technical specifications with reference URLs from official NVIDIA sources or reputable technology publications:

1. The official name of the platform announced at GTC 2026
2. The memory bandwidth per GPU for the Rubin GPU component (in TB/s)
3. The type of memory technology used in the Rubin GPU
4. For the NVL72 rack-scale configuration:
   - Total memory bandwidth (in TB/s)
   - NVLink interconnect bandwidth (in TB/s)

For each specification provided, include at least one reference URL that supports the information.
"""

# Ground-truth expectations used for strict checks on the answer text
GT = {
    "platform_official_name": "Vera Rubin",
    "rubin_mem_bw_tbps": "3.6 TB/s",
    "rubin_mem_type": "HBM4",
    "nvl72_total_bw_tbps": "1,580 TB/s",
    "nvl72_nvlink_bw_tbps": "260 TB/s",
}

QUALIFYING_SOURCE_INSTRUCTION = (
    "Only treat a source as valid if it is either: "
    "(a) an official NVIDIA domain (e.g., nvidia.com, blogs.nvidia.com, developer.nvidia.com), or "
    "(b) a reputable technology publication (e.g., anandtech.com, tomshardware.com, arstechnica.com, theverge.com, "
    "ieee.org, semianalysis.com, nextplatform.com, techcrunch.com, wired.com). "
    "If none of the provided URLs fit this criterion, the claim is NOT supported."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class GTC2026Extraction(BaseModel):
    """
    Extract exactly what the answer states plus the URLs it cites for each requested spec.
    All fields should reflect the answer text verbatim (do NOT invent or normalize).
    """
    # Platform name and its supporting URLs
    platform_name: Optional[str] = None
    platform_name_urls: List[str] = Field(default_factory=list)

    # Announcement support URLs (GTC 2026 event/dates/platform announcement)
    announcement_urls: List[str] = Field(default_factory=list)

    # Rubin GPU: memory bandwidth per GPU + URLs
    rubin_memory_bandwidth_per_gpu: Optional[str] = None
    rubin_memory_bandwidth_urls: List[str] = Field(default_factory=list)

    # Rubin GPU: memory technology type + URLs
    rubin_memory_type: Optional[str] = None
    rubin_memory_type_urls: List[str] = Field(default_factory=list)

    # NVL72: total memory bandwidth + URLs
    nvl72_total_memory_bandwidth: Optional[str] = None
    nvl72_total_memory_bandwidth_urls: List[str] = Field(default_factory=list)

    # NVL72: NVLink interconnect bandwidth + URLs
    nvl72_nvlink_bandwidth: Optional[str] = None
    nvl72_nvlink_bandwidth_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_gtc2026() -> str:
    return """
Extract the following fields strictly from the provided answer text. Do NOT infer or invent any values. If the answer does not provide a field, set it to null (for strings) or an empty array (for URL lists).

Required fields:
1) platform_name: The official platform name as stated in the answer (string).
2) platform_name_urls: All URLs the answer cites that specifically support the platform name (array of URLs).

3) announcement_urls: All URLs the answer cites that support the claim that the platform was announced at NVIDIA GTC 2026 and/or that GTC 2026 occurred March 16–19, 2026 (array of URLs).

4) rubin_memory_bandwidth_per_gpu: The Rubin GPU memory bandwidth per GPU exactly as stated in the answer (string, keep units as written).
5) rubin_memory_bandwidth_urls: All URLs the answer cites to support that Rubin GPU per‑GPU memory bandwidth (array of URLs).

6) rubin_memory_type: The memory technology used in the Rubin GPU exactly as stated in the answer (string, e.g., "HBM4" if the answer says so).
7) rubin_memory_type_urls: All URLs the answer cites to support the Rubin GPU memory type (array of URLs).

8) nvl72_total_memory_bandwidth: The NVL72 total memory bandwidth exactly as stated in the answer (string, keep units as written).
9) nvl72_total_memory_bandwidth_urls: All URLs the answer cites to support the NVL72 total memory bandwidth (array of URLs).

10) nvl72_nvlink_bandwidth: The NVL72 NVLink interconnect bandwidth exactly as stated in the answer (string, keep units as written).
11) nvl72_nvlink_bandwidth_urls: All URLs the answer cites to support the NVL72 NVLink bandwidth (array of URLs).

Rules for URL extraction:
- Extract only URLs that are explicitly present in the answer text (plain links or markdown).
- Do not include invalid or malformed URLs.
- If a URL is missing a protocol, prepend http://
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_announcement_section(evaluator: Evaluator, parent_node, ex: GTC2026Extraction) -> None:
    """
    Build and verify the 'Announcement_At_GTC_2026' subtree:
    - Announcement_Event_And_Dates_Stated (answer content check)
    - Announcement_URLs_Provided (existence)
    - Announcement_Supported_By_URL (evidence check)
    """
    ann_node = evaluator.add_parallel(
        id="Announcement_At_GTC_2026",
        desc="States that the platform was announced at NVIDIA GTC 2026 (March 16–19, 2026) and provides supporting reference URL(s).",
        parent=parent_node,
        critical=True
    )

    # 1) Check the answer explicitly states the announcement at GTC 2026 with the date range
    stated_leaf = evaluator.add_leaf(
        id="Announcement_Event_And_Dates_Stated",
        desc="Explicitly states the platform was announced at NVIDIA GTC 2026 (March 16–19, 2026).",
        parent=ann_node,
        critical=True
    )
    claim_stated = (
        "In the answer text, it is explicitly stated that the platform was announced at NVIDIA GTC 2026, "
        "and the date range March 16–19, 2026 is explicitly mentioned (allow hyphen/en-dash variants)."
    )
    await evaluator.verify(
        claim=claim_stated,
        node=stated_leaf,
        additional_instruction="Check only the answer text. Do not use outside knowledge. Accept minor formatting variants like 'Mar' vs 'March' and hyphen/en‑dash between 16 and 19."
    )

    # 2) Ensure at least one URL is provided for the announcement/dates claim
    evaluator.add_custom_node(
        result=bool(ex.announcement_urls),
        id="Announcement_URLs_Provided",
        desc="At least one qualifying reference URL is provided for the GTC 2026 announcement/dates claim.",
        parent=ann_node,
        critical=True
    )

    # 3) Verify those URLs support the announcement/dates claim
    supported_leaf = evaluator.add_leaf(
        id="Announcement_Supported_By_URL",
        desc="Includes at least one reference URL supporting the GTC 2026 announcement/dates claim.",
        parent=ann_node,
        critical=True
    )
    await evaluator.verify(
        claim="NVIDIA announced the next-generation AI platform at GTC 2026, which took place March 16–19, 2026.",
        node=supported_leaf,
        sources=ex.announcement_urls,
        additional_instruction=(
            "Confirm the page(s) explicitly mention GTC 2026 and/or the platform announcement at GTC 2026, "
            "including the dates March 16–19, 2026 if available. "
            + QUALIFYING_SOURCE_INSTRUCTION
        )
    )


async def verify_platform_name_section(evaluator: Evaluator, parent_node, ex: GTC2026Extraction) -> None:
    node = evaluator.add_parallel(
        id="Platform_Official_Name",
        desc="Provides the official name of the platform announced at GTC 2026, with supporting reference URL(s).",
        parent=parent_node,
        critical=True
    )

    # 1) Check the answer states exactly "Vera Rubin" as the platform name
    equals_leaf = evaluator.add_leaf(
        id="Platform_Name_Equals_Vera_Rubin",
        desc="States the platform name exactly as “Vera Rubin”.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer text, the official platform name is exactly 'Vera Rubin' (case-insensitive, ignore surrounding whitespace).",
        node=equals_leaf,
        additional_instruction="Check only the answer text. The key phrase must be 'Vera Rubin'. Do not accept 'Rubin' alone without 'Vera'."
    )

    # 2) Ensure at least one URL is provided for the platform name
    evaluator.add_custom_node(
        result=bool(ex.platform_name_urls),
        id="Platform_Name_URLs_Provided",
        desc="At least one qualifying reference URL is provided to support the platform name.",
        parent=node,
        critical=True
    )

    # 3) URLs support the platform name as "Vera Rubin"
    supported_leaf = evaluator.add_leaf(
        id="Platform_Name_Supported_By_URL",
        desc="Includes at least one reference URL that supports the stated platform name.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The official name of the platform announced at GTC 2026 is 'Vera Rubin'.",
        node=supported_leaf,
        sources=ex.platform_name_urls,
        additional_instruction=("Verify that the page explicitly names the platform as 'Vera Rubin'. " + QUALIFYING_SOURCE_INSTRUCTION)
    )


async def verify_rubin_mem_bandwidth_section(evaluator: Evaluator, parent_node, ex: GTC2026Extraction) -> None:
    node = evaluator.add_parallel(
        id="Rubin_GPU_Memory_Bandwidth_per_GPU",
        desc="Provides the memory bandwidth per GPU for the Rubin GPU component (in TB/s), with supporting reference URL(s).",
        parent=parent_node,
        critical=True
    )

    # 1) Check the answer states 3.6 TB/s (or clear equivalent like 3600 GB/s)
    equals_leaf = evaluator.add_leaf(
        id="Bandwidth_Equals_3_6_TBps",
        desc="States Rubin GPU memory bandwidth per GPU as 3.6 TB/s (or an equivalent value that clearly converts to 3.6 TB/s).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer text, the Rubin GPU memory bandwidth per GPU is stated as 3.6 TB/s (or an equivalent such as 3600 GB/s).",
        node=equals_leaf,
        additional_instruction="Check only the answer text. Treat common equivalents as acceptable: 3.6 TB/s == 3600 GB/s."
    )

    # 2) Ensure at least one URL is provided
    evaluator.add_custom_node(
        result=bool(ex.rubin_memory_bandwidth_urls),
        id="Bandwidth_URLs_Provided",
        desc="At least one qualifying reference URL is provided for the Rubin per‑GPU memory bandwidth.",
        parent=node,
        critical=True
    )

    # 3) URLs support the 3.6 TB/s spec
    supported_leaf = evaluator.add_leaf(
        id="Bandwidth_Supported_By_URL",
        desc="Includes at least one reference URL that supports the stated memory bandwidth per GPU.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The memory bandwidth per Rubin GPU is 3.6 TB/s (equivalently 3600 GB/s).",
        node=supported_leaf,
        sources=ex.rubin_memory_bandwidth_urls,
        additional_instruction=("Confirm the page states 3.6 TB/s (or a clear equivalent like 3600 GB/s) per Rubin GPU. " + QUALIFYING_SOURCE_INSTRUCTION)
    )


async def verify_rubin_mem_type_section(evaluator: Evaluator, parent_node, ex: GTC2026Extraction) -> None:
    node = evaluator.add_parallel(
        id="Rubin_GPU_Memory_Technology_Type",
        desc="Provides the type of memory technology used in the Rubin GPU, with supporting reference URL(s).",
        parent=parent_node,
        critical=True
    )

    # 1) Check the answer states HBM4
    equals_leaf = evaluator.add_leaf(
        id="Memory_Technology_Equals_HBM4",
        desc="States the Rubin GPU memory technology type as HBM4.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer text, the Rubin GPU's memory technology is stated as HBM4.",
        node=equals_leaf,
        additional_instruction="Check only the answer text. Accept case-insensitive 'HBM4'. Treat 'HBM4e' as part of the HBM4 generation only if the answer also clearly references HBM4."
    )

    # 2) Ensure at least one URL provided
    evaluator.add_custom_node(
        result=bool(ex.rubin_memory_type_urls),
        id="Memory_Type_URLs_Provided",
        desc="At least one qualifying reference URL is provided for the Rubin memory technology type.",
        parent=node,
        critical=True
    )

    # 3) URLs support HBM4
    supported_leaf = evaluator.add_leaf(
        id="Memory_Type_Supported_By_URL",
        desc="Includes at least one reference URL that supports the stated memory technology type.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Rubin GPU uses HBM4 memory technology.",
        node=supported_leaf,
        sources=ex.rubin_memory_type_urls,
        additional_instruction=("Confirm the page explicitly indicates HBM4 for Rubin GPU. " + QUALIFYING_SOURCE_INSTRUCTION)
    )


async def verify_nvl72_section(evaluator: Evaluator, parent_node, ex: GTC2026Extraction) -> None:
    nvl_node = evaluator.add_parallel(
        id="NVL72_Rack_Scale_Configuration_Specs",
        desc="For the NVL72 rack-scale configuration, provides total memory bandwidth and NVLink interconnect bandwidth (both in TB/s), each with supporting reference URL(s).",
        parent=parent_node,
        critical=True
    )

    # ---- NVL72 Total Memory Bandwidth ----
    total_node = evaluator.add_parallel(
        id="NVL72_Total_Memory_Bandwidth",
        desc="Provides NVL72 total memory bandwidth (in TB/s), with supporting reference URL(s).",
        parent=nvl_node,
        critical=True
    )

    total_equals_leaf = evaluator.add_leaf(
        id="NVL72_Total_Bandwidth_Equals_1580_TBps",
        desc="States NVL72 total memory bandwidth as 1,580 TB/s (or an equivalent value that clearly converts to 1,580 TB/s).",
        parent=total_node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer text, the NVL72 total memory bandwidth is stated as 1,580 TB/s (equivalently 1.58 PB/s).",
        node=total_equals_leaf,
        additional_instruction="Check only the answer text. Accept numeric formatting variants like '1580 TB/s' or '1.58 PB/s'."
    )

    evaluator.add_custom_node(
        result=bool(ex.nvl72_total_memory_bandwidth_urls),
        id="NVL72_Total_BW_URLs_Provided",
        desc="At least one qualifying reference URL is provided for the NVL72 total memory bandwidth.",
        parent=total_node,
        critical=True
    )

    total_supported_leaf = evaluator.add_leaf(
        id="NVL72_Total_Bandwidth_Supported_By_URL",
        desc="Includes at least one reference URL that supports the stated NVL72 total memory bandwidth.",
        parent=total_node,
        critical=True
    )
    await evaluator.verify(
        claim="The NVL72 total memory bandwidth is 1,580 TB/s (equivalently 1.58 PB/s).",
        node=total_supported_leaf,
        sources=ex.nvl72_total_memory_bandwidth_urls,
        additional_instruction=("Confirm the page states (or clearly implies) total memory bandwidth of 1,580 TB/s for NVL72. " + QUALIFYING_SOURCE_INSTRUCTION)
    )

    # ---- NVL72 NVLink Interconnect Bandwidth ----
    nvlink_node = evaluator.add_parallel(
        id="NVL72_NVLink_Interconnect_Bandwidth",
        desc="Provides NVL72 NVLink interconnect bandwidth (in TB/s), with supporting reference URL(s).",
        parent=nvl_node,
        critical=True
    )

    nvlink_equals_leaf = evaluator.add_leaf(
        id="NVL72_NVLink_Bandwidth_Equals_260_TBps",
        desc="States NVL72 NVLink interconnect bandwidth as 260 TB/s (or an equivalent value that clearly converts to 260 TB/s).",
        parent=nvlink_node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer text, the NVL72 NVLink interconnect bandwidth is stated as 260 TB/s.",
        node=nvlink_equals_leaf,
        additional_instruction="Check only the answer text. Accept case and minor formatting variants; the numeric value must be 260 TB/s."
    )

    evaluator.add_custom_node(
        result=bool(ex.nvl72_nvlink_bandwidth_urls),
        id="NVL72_NVLink_BW_URLs_Provided",
        desc="At least one qualifying reference URL is provided for the NVL72 NVLink interconnect bandwidth.",
        parent=nvlink_node,
        critical=True
    )

    nvlink_supported_leaf = evaluator.add_leaf(
        id="NVL72_NVLink_Bandwidth_Supported_By_URL",
        desc="Includes at least one reference URL that supports the stated NVL72 NVLink interconnect bandwidth.",
        parent=nvlink_node,
        critical=True
    )
    await evaluator.verify(
        claim="The NVL72 NVLink interconnect bandwidth is 260 TB/s.",
        node=nvlink_supported_leaf,
        sources=ex.nvl72_nvlink_bandwidth_urls,
        additional_instruction=("Confirm the page states (or clearly implies) 260 TB/s NVLink interconnect bandwidth for NVL72. " + QUALIFYING_SOURCE_INSTRUCTION)
    )


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the NVIDIA GTC 2026 next-gen platform and specs task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level rubric node is parallel
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

    # Extract structured data from the answer
    extraction: GTC2026Extraction = await evaluator.extract(
        prompt=prompt_extract_gtc2026(),
        template_class=GTC2026Extraction,
        extraction_name="gtc_2026_extraction",
    )

    # Add ground truth expectations for transparency
    evaluator.add_ground_truth(
        {
            "expected_platform_official_name": GT["platform_official_name"],
            "expected_rubin_per_gpu_bw_tbps": GT["rubin_mem_bw_tbps"],
            "expected_rubin_memory_type": GT["rubin_mem_type"],
            "expected_nvl72_total_bw_tbps": GT["nvl72_total_bw_tbps"],
            "expected_nvl72_nvlink_bw_tbps": GT["nvl72_nvlink_bw_tbps"],
        },
        gt_type="expected_specs",
    )

    # Build rubric tree main node (critical)
    main_node = evaluator.add_parallel(
        id="GTC_2026_NextGen_Platform_and_Specs",
        desc="Identify the next-generation AI computing platform unveiled at NVIDIA GTC 2026 and provide the requested technical specifications, each supported by at least one qualifying reference URL.",
        parent=root,
        critical=True,
    )

    # Subsections
    await verify_announcement_section(evaluator, main_node, extraction)
    await verify_platform_name_section(evaluator, main_node, extraction)
    await verify_rubin_mem_bandwidth_section(evaluator, main_node, extraction)
    await verify_rubin_mem_type_section(evaluator, main_node, extraction)
    await verify_nvl72_section(evaluator, main_node, extraction)

    # Return final summary
    return evaluator.get_summary()