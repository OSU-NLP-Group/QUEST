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
TASK_ID = "ai_accelerator_selection"
TASK_DESCRIPTION = (
    "Identify an AI accelerator chip suitable for enterprise data center deployment in 2025-2026 that meets all of the following technical requirements: "
    "(1) Memory capacity of at least 128GB, "
    "(2) Memory bandwidth of at least 3.5 TB/s, "
    "(3) Maximum power consumption not exceeding 800W, "
    "(4) Uses High Bandwidth Memory (HBM) technology, specifically HBM2e or HBM3, "
    "(5) Manufactured by one of the established AI accelerator companies: AMD, Intel, Google, AWS, or NVIDIA, "
    "(6) From an architecture generation released in 2023 or later and actively used in 2024-2025 data centers, "
    "(7) Has publicly verifiable specifications from official manufacturer product pages or documentation. "
    "Provide the chip name, manufacturer, and the official product page URL where all these specifications can be verified."
)

ALLOWED_MANUFACTURERS = ["AMD", "Intel", "Google", "AWS", "NVIDIA"]
ALLOWED_MANUFACTURER_DOMAINS = [
    "amd.com",
    "intel.com",
    "nvidia.com",
    "developer.nvidia.com",
    "docs.nvidia.com",
    "google.com",
    "cloud.google.com",
    "ai.google",
    "aws.amazon.com",
]

# Thresholds
MIN_MEMORY_CAPACITY_GB = 128
MIN_MEMORY_BANDWIDTH_TBPS = 3.5
MAX_BOARD_POWER_W = 800

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ChipInfo(BaseModel):
    chip_name: Optional[str] = None
    manufacturer: Optional[str] = None
    official_url: Optional[str] = None

    # Specs as stated in the answer (strings preferred for flexibility)
    memory_capacity: Optional[str] = None        # e.g., "192 GB HBM3"
    memory_bandwidth: Optional[str] = None       # e.g., "4.8 TB/s"
    power_consumption: Optional[str] = None      # e.g., "700W TBP"
    memory_technology: Optional[str] = None      # e.g., "HBM3e"
    architecture_generation: Optional[str] = None  # e.g., "Hopper", "MI300"
    release_year_or_date: Optional[str] = None     # e.g., "2023", "Nov 2023"
    datacenter_usage_notes_2024_2025: Optional[str] = None  # any claim in answer about usage in 2024-2025


class ChipSelectionExtraction(BaseModel):
    selected_chip: Optional[ChipInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_chip_selection() -> str:
    return (
        "Extract the single AI accelerator chip proposed in the answer that is claimed to meet the requirements. "
        "Return a JSON object under 'selected_chip' with fields:\n"
        "1) chip_name: name of the chip (e.g., 'NVIDIA H200', 'AMD Instinct MI300X').\n"
        "2) manufacturer: company name (e.g., 'NVIDIA', 'AMD', 'Intel', 'Google', 'AWS').\n"
        "3) official_url: the official manufacturer product page or documentation URL where specs can be verified. "
        "   Only include a valid full URL explicitly present in the answer.\n"
        "4) memory_capacity: the device memory capacity as stated in the answer (string; include units if present).\n"
        "5) memory_bandwidth: the memory bandwidth as stated in the answer (string; include units if present).\n"
        "6) power_consumption: the typical board power or max power consumption as stated in the answer (string; include units if present).\n"
        "7) memory_technology: the memory technology stated (e.g., 'HBM3', 'HBM3e', 'HBM2e').\n"
        "8) architecture_generation: the architecture generation stated (e.g., 'Hopper', 'Blackwell', 'MI300').\n"
        "9) release_year_or_date: release or announcement year/date if stated (string).\n"
        "10) datacenter_usage_notes_2024_2025: any claim in the answer about the chip being used/deployed in data centers in 2024-2025.\n\n"
        "Rules:\n"
        "- Extract only what is explicitly stated in the answer; if a field is missing, set it to null.\n"
        "- For URLs, extract the actual URL string; if missing protocol, prepend 'http://'.\n"
        "- Do not infer or invent values."
    )


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def verify_chip_selection(
    evaluator: Evaluator,
    parent_node,
    chip: ChipInfo,
) -> None:
    """
    Build and verify the rubric tree for the chip selection.

    We verify the official documentation first (critical sibling) to gate subsequent checks.
    """
    # Create the critical parallel aggregator as specified in the rubric
    selection_node = evaluator.add_parallel(
        id="AI_Accelerator_Selection",
        desc="Evaluate whether the identified AI accelerator chip meets all specified technical requirements for enterprise data center deployment",
        parent=parent_node,
        critical=True
    )

    # Official documentation check (critical)
    official_doc_leaf = evaluator.add_leaf(
        id="Official_Documentation_Check",
        desc="All technical specifications must be verifiable from official manufacturer product pages or documentation",
        parent=selection_node,
        critical=True
    )
    official_doc_claim = (
        f"The provided URL is an official manufacturer product page or official documentation for the chip "
        f"and it publicly lists technical specifications relevant to memory capacity, bandwidth, power, and memory technology."
    )
    await evaluator.verify(
        claim=official_doc_claim,
        node=official_doc_leaf,
        sources=chip.official_url,
        additional_instruction=(
            "Verify that the URL belongs to an official manufacturer domain and provides technical specifications. "
            "Examples of official domains include: amd.com, intel.com, nvidia.com, developer.nvidia.com, docs.nvidia.com, "
            "google.com, cloud.google.com, ai.google, aws.amazon.com. "
            "If no URL is provided or the page is not from an official manufacturer site or does not list specs, mark as not supported."
        ),
    )

    # Prepare the remaining verifications; ensure they are critical and use the official URL
    # Memory capacity check (critical)
    mem_cap_leaf = evaluator.add_leaf(
        id="Memory_Capacity_Check",
        desc="The chip must have GPU/device memory capacity of at least 128GB to handle large-scale AI model training and inference workloads",
        parent=selection_node,
        critical=True
    )
    mem_cap_claim = (
        f"The official product page indicates the total device memory capacity for the chip is at least {MIN_MEMORY_CAPACITY_GB} GB."
    )
    mem_cap_add_ins = (
        "Look for total device memory capacity (HBM capacity per device). "
        "Accept minor unit variants (GB vs GiB). "
        "If multiple configurations are listed, use the highest capacity configuration. "
        "If the page states a capacity like 141 GB, 192 GB, or 256 GB, that satisfies 'at least 128 GB'. "
        "If the page only shows per-stack capacity, ensure the total per device is >= 128 GB."
    )

    # Memory bandwidth check (critical)
    mem_bw_leaf = evaluator.add_leaf(
        id="Memory_Bandwidth_Check",
        desc="The chip must provide memory bandwidth of at least 3.5 TB/s to ensure sufficient data throughput for AI computations",
        parent=selection_node,
        critical=True
    )
    mem_bw_claim = (
        f"The official product page indicates memory bandwidth for the device is at least {MIN_MEMORY_BANDWIDTH_TBPS} TB/s."
    )
    mem_bw_add_ins = (
        "Use the device-level memory bandwidth (aggregate across HBM stacks if the page specifies aggregate bandwidth). "
        "Accept phrasing such as 'up to' or 'peak' bandwidth, as long as it meets or exceeds the threshold."
    )

    # Power consumption check (critical)
    power_leaf = evaluator.add_leaf(
        id="Power_Consumption_Check",
        desc="The chip's typical board power or maximum power consumption must not exceed 800W to comply with data center thermal management constraints",
        parent=selection_node,
        critical=True
    )
    power_claim = (
        f"The official product page indicates the device's typical board power or maximum power (TBP/TDP) does not exceed {MAX_BOARD_POWER_W} W."
    )
    power_add_ins = (
        "Check TBP, TDP, or maximum power for the accelerator card or module. "
        "If multiple configurations are listed, use the highest stated configuration and verify it is <= 800 W. "
        "If only a single value is listed (e.g., 700 W), that satisfies the constraint."
    )

    # Memory technology check (critical)
    mem_tech_leaf = evaluator.add_leaf(
        id="Memory_Technology_Check",
        desc="The chip must use High Bandwidth Memory (HBM) technology, specifically HBM2e or HBM3, rather than GDDR memory",
        parent=selection_node,
        critical=True
    )
    mem_tech_claim = (
        "The official product page indicates the device uses HBM2e or HBM3 memory technology (HBM3e counts as HBM3)."
    )
    mem_tech_add_ins = (
        "If the page states 'HBM3e', consider it a subset of HBM3 and therefore acceptable. "
        "Reject technologies such as GDDR or plain DDR as they do not meet the requirement."
    )

    # Manufacturer verification (critical)
    mfg_leaf = evaluator.add_leaf(
        id="Manufacturer_Verification",
        desc="The chip must be manufactured by an established AI accelerator company among: AMD, Intel, Google, AWS, or NVIDIA",
        parent=selection_node,
        critical=True
    )
    mfg_claim = (
        "This product page is for an accelerator manufactured by one of: AMD, Intel, Google, AWS, or NVIDIA."
    )
    mfg_add_ins = (
        "Confirm manufacturer branding and product lineage on the page. "
        "You may use the domain and page content to determine the manufacturer."
    )

    # Architecture generation check (critical)
    arch_leaf = evaluator.add_leaf(
        id="Architecture_Generation_Check",
        desc="The chip must be from an architecture generation released in 2023 or later and actively used in 2024-2025 data centers to ensure support for modern AI workload requirements",
        parent=selection_node,
        critical=True
    )
    arch_claim = (
        "The official documentation indicates the architecture generation was introduced/released in 2023 or later and that the accelerator is used or deployed in data centers during 2024-2025."
    )
    arch_add_ins = (
        "Look for evidence of release or introduction in 2023 or later (e.g., 'announced November 2023', 'launched 2024'). "
        "Also look for statements indicating datacenter usage or deployment in 2024-2025 (e.g., 'for data centers', 'shipping in 2024', 'deployed by cloud providers', 'in production for 2024/2025'). "
        "If the official page clearly targets datacenters and indicates availability/production in 2024 or 2025, consider this satisfied."
    )

    # Verify the remaining checks, after official documentation check
    claims_and_sources = [
        (mem_cap_claim, chip.official_url, mem_cap_leaf, mem_cap_add_ins),
        (mem_bw_claim, chip.official_url, mem_bw_leaf, mem_bw_add_ins),
        (power_claim, chip.official_url, power_leaf, power_add_ins),
        (mem_tech_claim, chip.official_url, mem_tech_leaf, mem_tech_add_ins),
        (mfg_claim, chip.official_url, mfg_leaf, mfg_add_ins),
        (arch_claim, chip.official_url, arch_leaf, arch_add_ins),
    ]
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for AI accelerator selection suitability against the rubric.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # The rubric's top node is parallel
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

    # Extract chip selection info
    extraction = await evaluator.extract(
        prompt=prompt_extract_chip_selection(),
        template_class=ChipSelectionExtraction,
        extraction_name="chip_selection_extraction",
    )
    chip = extraction.selected_chip or ChipInfo()

    # Record ground truth thresholds and allowed manufacturers for transparency
    evaluator.add_ground_truth({
        "minimum_requirements": {
            "memory_capacity_gb_min": MIN_MEMORY_CAPACITY_GB,
            "memory_bandwidth_tbps_min": MIN_MEMORY_BANDWIDTH_TBPS,
            "power_w_max": MAX_BOARD_POWER_W,
            "memory_technology_allowed": ["HBM2e", "HBM3", "HBM3e (counts as HBM3)"],
            "allowed_manufacturers": ALLOWED_MANUFACTURERS,
            "release_year_min": 2023,
            "datacenter_usage_window": "2024-2025"
        }
    })

    # Add custom info to help debugging
    evaluator.add_custom_info(
        info={
            "extracted_chip_name": chip.chip_name,
            "extracted_manufacturer": chip.manufacturer,
            "official_url": chip.official_url,
            "answer_memory_capacity": chip.memory_capacity,
            "answer_memory_bandwidth": chip.memory_bandwidth,
            "answer_power_consumption": chip.power_consumption,
            "answer_memory_technology": chip.memory_technology,
            "answer_architecture_generation": chip.architecture_generation,
            "answer_release_year_or_date": chip.release_year_or_date,
            "answer_datacenter_usage_notes_2024_2025": chip.datacenter_usage_notes_2024_2025,
        },
        info_type="extraction_debug"
    )

    # Build and run verification according to rubric
    await verify_chip_selection(evaluator, root, chip)

    # Return structured evaluation summary
    return evaluator.get_summary()