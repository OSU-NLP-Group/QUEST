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
TASK_ID = "apple_tablets_march_2025_m_series_3nm"
TASK_DESCRIPTION = (
    "Identify all Apple tablet products that were announced in March 2025 and use Apple M-series processors "
    "manufactured with 3nm fabrication process technology. For each identified tablet model (considering different "
    "screen sizes as separate models), provide comprehensive technical specifications including: "
    "(1) Product Identification: Product name, exact announcement date, and release date; "
    "(2) Processor Architecture: Complete details including total CPU core count, number of performance cores, number of efficiency cores, GPU core count, Neural Engine specifications, and support for hardware-accelerated ray tracing; "
    "(3) Memory Configuration: RAM capacity in gigabytes; "
    "(4) Storage Options: All available storage capacity tiers (128GB, 256GB, 512GB, 1TB if applicable); "
    "(5) Display Specifications: Screen size in inches, display resolution (width × height in pixels), maximum brightness in nits, and display technology type; "
    "(6) Connectivity Standards: Wi-Fi standard version and Bluetooth version; "
    "(7) Battery Capacity: Battery capacity in watt-hours (Wh). "
    "Each specification must include a reference URL from official Apple sources (apple.com or support.apple.com) that verifies the provided information."
)

# Expected constraints per rubric
ANNOUNCEMENT_DATE_STR = "March 4, 2025"
RELEASE_DATE_STR = "March 12, 2025"
PROCESS_NODE = "3nm"

# Processor architecture constraints (per rubric)
CPU_TOTAL = 8
CPU_PERF = 4
CPU_EFF = 4
GPU_CORES = 9
NE_CORES = 16
RAY_TRACING_SUPPORTED = True

# Memory constraints
RAM_GB = 8

# Storage constraints
STORAGE_TIERS = ["128GB", "256GB", "512GB", "1TB"]

# Display constraints (distinct by model)
DISPLAY_11_TECH = "Liquid Retina"
DISPLAY_11_RES = "2360×1640"
DISPLAY_11_BRIGHTNESS = 500

DISPLAY_13_TECH = "Liquid Retina"
DISPLAY_13_RES = "2732×2048"
DISPLAY_13_BRIGHTNESS = 600

# Connectivity constraints
WIFI_STD = "Wi‑Fi 6E (802.11ax)"
BT_VERSION = "5.3"

# Battery capacity constraints
BATTERY_WH_11 = 28.93
BATTERY_WH_13 = 36.59


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ModelSpec(BaseModel):
    # Identification
    product_name: Optional[str] = None
    device_type: Optional[str] = None
    announcement_date: Optional[str] = None
    release_date: Optional[str] = None
    # Processor
    m_series_name: Optional[str] = None  # e.g., "M3"
    process_node_nm: Optional[str] = None  # e.g., "3nm"
    cpu_total_cores: Optional[str] = None
    cpu_performance_cores: Optional[str] = None
    cpu_efficiency_cores: Optional[str] = None
    gpu_cores: Optional[str] = None
    neural_engine_cores: Optional[str] = None
    hardware_ray_tracing_supported: Optional[str] = None  # "yes"/"no"/"supported"/"not supported"
    # Memory
    memory_ram_gb: Optional[str] = None
    # Storage
    storage_options: List[str] = Field(default_factory=list)  # e.g., ["128GB","256GB","512GB","1TB"]
    # Display
    screen_size_inch: Optional[str] = None  # e.g., "11-inch" or "13-inch"
    display_technology: Optional[str] = None
    display_resolution_w: Optional[str] = None
    display_resolution_h: Optional[str] = None
    display_resolution_str: Optional[str] = None  # e.g., "2360×1640"
    display_brightness_nits: Optional[str] = None
    # Connectivity
    wifi_standard: Optional[str] = None  # e.g., "Wi‑Fi 6E (802.11ax)"
    bluetooth_version: Optional[str] = None  # e.g., "5.3"
    # Battery
    battery_capacity_wh: Optional[str] = None

    # Official references per section
    identification_urls: List[str] = Field(default_factory=list)
    processor_urls: List[str] = Field(default_factory=list)
    memory_urls: List[str] = Field(default_factory=list)
    storage_urls: List[str] = Field(default_factory=list)
    display_urls: List[str] = Field(default_factory=list)
    connectivity_urls: List[str] = Field(default_factory=list)
    battery_urls: List[str] = Field(default_factory=list)


class TabletModelsExtraction(BaseModel):
    models: List[ModelSpec] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_models() -> str:
    return """
    From the provided answer, extract all Apple tablet models (iPad family) that the answer claims were announced in March 2025 and use Apple M‑series processors with a 3nm fabrication process. Treat different screen sizes (e.g., 11‑inch vs 13‑inch) as separate models.

    For each model, extract the following fields exactly as stated in the answer. If a field is missing, return null or an empty list as appropriate.

    Identification:
    - product_name
    - device_type (e.g., "iPad", "tablet")
    - announcement_date (e.g., "March 4, 2025")
    - release_date (e.g., "March 12, 2025")
    - identification_urls (official Apple URLs from apple.com or support.apple.com that the answer cites for identification/announcement/release info)

    Processor:
    - m_series_name (e.g., "M3")
    - process_node_nm (e.g., "3nm")
    - cpu_total_cores (e.g., "8")
    - cpu_performance_cores (e.g., "4")
    - cpu_efficiency_cores (e.g., "4")
    - gpu_cores (e.g., "9")
    - neural_engine_cores (e.g., "16")
    - hardware_ray_tracing_supported (e.g., "yes"/"no"/"supported"/"not supported")
    - processor_urls (official Apple URLs that the answer cites for processor architecture specs)

    Memory:
    - memory_ram_gb (e.g., "8 GB")
    - memory_urls (official Apple URLs that the answer cites for RAM capacity)

    Storage:
    - storage_options (list of strings like "128GB","256GB","512GB","1TB")
    - storage_urls (official Apple URLs that the answer cites for storage tiers)

    Display:
    - screen_size_inch (e.g., "11-inch" or "13-inch")
    - display_technology (e.g., "Liquid Retina")
    - display_resolution_w (e.g., "2360")
    - display_resolution_h (e.g., "1640")
    - display_resolution_str (e.g., "2360×1640"; include the '×' if present in the answer)
    - display_brightness_nits (e.g., "500")
    - display_urls (official Apple URLs that the answer cites for display specs)

    Connectivity:
    - wifi_standard (e.g., "Wi‑Fi 6E (802.11ax)")
    - bluetooth_version (e.g., "5.3")
    - connectivity_urls (official Apple URLs that the answer cites for Wi‑Fi and Bluetooth specs)

    Battery:
    - battery_capacity_wh (e.g., "28.93 Wh")
    - battery_urls (official Apple URLs that the answer cites for battery capacity)

    SPECIAL URL RULES:
    - Only include URLs explicitly present in the answer. Extract the actual URLs, not just "Apple" mentions.
    - Prefer apple.com or support.apple.com links. If the answer provides non-Apple links, include them, but keep Apple URLs when present.
    - If any section-specific URLs are missing, return an empty list for that section.

    Return JSON:
    { "models": [ { ...fields above for each model... } ] }
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def has_official_apple_url(urls: List[str]) -> bool:
    if not urls:
        return False
    domain_markers = ["apple.com", "support.apple.com"]
    for u in urls:
        uu = (u or "").lower()
        if any(dm in uu for dm in domain_markers):
            return True
    return False


def pick_model_for_size(models: List[ModelSpec], target_size_inch: float) -> Optional[ModelSpec]:
    """Pick the first model whose screen_size_inch mentions the target size (e.g., '11' or '13')."""
    t_str = str(int(target_size_inch))
    for m in models:
        if m and m.screen_size_inch and t_str in m.screen_size_inch:
            return m
        # Also match in product name if size isn't in screen_size field
        if m and m.product_name and t_str in m.product_name:
            return m
    return None


def safe_sources(primary: List[str], fallback: List[str]) -> List[str]:
    """Prefer primary list if non-empty; otherwise, use fallback list."""
    if primary and len(primary) > 0:
        return primary
    return fallback or []


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_identification_and_eligibility(
    evaluator: Evaluator,
    parent_node,
    model_node_id: str,
    model_desc: str,
    model: Optional[ModelSpec],
    expected_screen_size_inch: float
) -> None:
    ident_node = evaluator.add_parallel(
        id=f"{model_node_id}_identification_and_eligibility",
        desc="Model identification and eligibility constraints",
        parent=parent_node,
        critical=True
    )

    # Custom existence checks
    product_name_ok = bool(model and model.product_name and model.product_name.strip())
    evaluator.add_custom_node(
        result=product_name_ok,
        id=f"{model_node_id}_product_name_provided",
        desc="Product name is provided",
        parent=ident_node,
        critical=True
    )

    ident_urls = model.identification_urls if model else []
    evaluator.add_custom_node(
        result=has_official_apple_url(ident_urls),
        id=f"{model_node_id}_official_reference_url_for_identification",
        desc="Provides at least one reference URL from apple.com or support.apple.com verifying identification/announcement/release information",
        parent=ident_node,
        critical=True
    )

    # Tablet device check (verify against Apple page)
    leaf_is_tablet = evaluator.add_leaf(
        id=f"{model_node_id}_is_tablet_device",
        desc="Device is a tablet product",
        parent=ident_node,
        critical=True
    )
    claim_is_tablet = f"This product is an iPad tablet device."
    await evaluator.verify(
        claim=claim_is_tablet,
        node=leaf_is_tablet,
        sources=ident_urls,
        additional_instruction="Confirm the product is part of the iPad lineup (tablet). Accept 'iPad' as tablet."
    )

    # Announcement date equals March 4, 2025
    leaf_ann = evaluator.add_leaf(
        id=f"{model_node_id}_announcement_date_is_march_4_2025",
        desc="Announcement date equals March 4, 2025",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The product was announced on {ANNOUNCEMENT_DATE_STR}.",
        node=leaf_ann,
        sources=ident_urls,
        additional_instruction="Verify on Apple's official newsroom or product page the announcement date equals March 4, 2025."
    )

    # Release date equals March 12, 2025
    leaf_rel = evaluator.add_leaf(
        id=f"{model_node_id}_release_date_is_march_12_2025",
        desc="Release date equals March 12, 2025",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The product release date (availability) is {RELEASE_DATE_STR}.",
        node=leaf_rel,
        sources=ident_urls,
        additional_instruction="Verify on Apple's official product/specs/newsroom page that availability/release aligns with March 12, 2025 (or equivalent phrasing like 'available starting March 12, 2025')."
    )

    # Uses Apple M-series processor
    leaf_m_series = evaluator.add_leaf(
        id=f"{model_node_id}_uses_apple_m_series_processor",
        desc="Uses an Apple M-series processor",
        parent=ident_node,
        critical=True
    )
    proc_urls_for_id = safe_sources(model.processor_urls if model else [], ident_urls)
    await evaluator.verify(
        claim="This model uses an Apple M‑series processor (e.g., M1/M2/M3/M4).",
        node=leaf_m_series,
        sources=proc_urls_for_id,
        additional_instruction="Confirm the page explicitly states the device uses an Apple M‑series SoC."
    )

    # Processor is 3nm
    leaf_3nm = evaluator.add_leaf(
        id=f"{model_node_id}_processor_is_3nm",
        desc="Processor is manufactured using 3nm fabrication process technology",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Apple M‑series processor in this model is manufactured using a 3nm fabrication process.",
        node=leaf_3nm,
        sources=proc_urls_for_id,
        additional_instruction="Look for '3nm' in the SoC/process description on Apple's official page."
    )

    # Screen size equals expected inches
    leaf_screen = evaluator.add_leaf(
        id=f"{model_node_id}_screen_size_is_{int(expected_screen_size_inch)}_inch",
        desc=f"Screen size equals {int(expected_screen_size_inch)} inches (treated as a distinct model)",
        parent=ident_node,
        critical=True
    )
    display_urls = model.display_urls if model else []
    await evaluator.verify(
        claim=f"The screen size is {int(expected_screen_size_inch)} inches.",
        node=leaf_screen,
        sources=display_urls if display_urls else ident_urls,
        additional_instruction="Check the display size reported on Apple's tech specs page for this model/variant."
    )


async def verify_processor_architecture(
    evaluator: Evaluator,
    parent_node,
    model_node_id: str,
    model: Optional[ModelSpec]
) -> None:
    proc_node = evaluator.add_parallel(
        id=f"{model_node_id}_processor_architecture",
        desc="Processor architecture specifications (must match constraints) with official reference",
        parent=parent_node,
        critical=True
    )

    proc_urls = model.processor_urls if model else []

    evaluator.add_custom_node(
        result=has_official_apple_url(proc_urls),
        id=f"{model_node_id}_official_reference_url_for_processor",
        desc="Provides an official Apple reference URL (apple.com or support.apple.com) verifying processor architecture specs",
        parent=proc_node,
        critical=True
    )

    # CPU total cores = 8
    leaf_cpu_total = evaluator.add_leaf(
        id=f"{model_node_id}_cpu_total_cores_is_8",
        desc="Total CPU core count equals 8",
        parent=proc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The CPU has {CPU_TOTAL} total cores.",
        node=leaf_cpu_total,
        sources=proc_urls,
        additional_instruction="Verify core count on Apple's official specs page; allow equivalent phrasing like '8‑core CPU'."
    )

    # Performance cores = 4
    leaf_cpu_perf = evaluator.add_leaf(
        id=f"{model_node_id}_cpu_performance_cores_is_4",
        desc="Performance core count equals 4",
        parent=proc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The CPU includes {CPU_PERF} performance cores.",
        node=leaf_cpu_perf,
        sources=proc_urls,
        additional_instruction="Confirm performance core count on Apple's official SoC architecture description."
    )

    # Efficiency cores = 4
    leaf_cpu_eff = evaluator.add_leaf(
        id=f"{model_node_id}_cpu_efficiency_cores_is_4",
        desc="Efficiency core count equals 4",
        parent=proc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The CPU includes {CPU_EFF} efficiency cores.",
        node=leaf_cpu_eff,
        sources=proc_urls,
        additional_instruction="Confirm efficiency core count on Apple's official SoC architecture description."
    )

    # GPU cores = 9
    leaf_gpu = evaluator.add_leaf(
        id=f"{model_node_id}_gpu_cores_is_9",
        desc="GPU core count equals 9",
        parent=proc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The GPU has {GPU_CORES} cores.",
        node=leaf_gpu,
        sources=proc_urls,
        additional_instruction="Verify the GPU core count; allow phrasing like '9‑core GPU'."
    )

    # Neural Engine cores = 16
    leaf_ne = evaluator.add_leaf(
        id=f"{model_node_id}_neural_engine_cores_is_16",
        desc="Neural Engine core count equals 16",
        parent=proc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Neural Engine has {NE_CORES} cores.",
        node=leaf_ne,
        sources=proc_urls,
        additional_instruction="Verify Neural Engine core count on Apple's specs page."
    )

    # Hardware-accelerated ray tracing supported
    leaf_rt = evaluator.add_leaf(
        id=f"{model_node_id}_hardware_accelerated_ray_tracing_supported",
        desc="Hardware-accelerated ray tracing support is affirmed",
        parent=proc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The GPU supports hardware‑accelerated ray tracing.",
        node=leaf_rt,
        sources=proc_urls,
        additional_instruction="Confirm the presence of hardware‑accelerated ray tracing in Apple's description."
    )


async def verify_memory(
    evaluator: Evaluator,
    parent_node,
    model_node_id: str,
    model: Optional[ModelSpec]
) -> None:
    mem_node = evaluator.add_parallel(
        id=f"{model_node_id}_memory_configuration",
        desc="Memory configuration with official reference",
        parent=parent_node,
        critical=True
    )

    mem_urls = model.memory_urls if model else []
    evaluator.add_custom_node(
        result=has_official_apple_url(mem_urls),
        id=f"{model_node_id}_official_reference_url_for_memory",
        desc="Provides an official Apple reference URL verifying RAM capacity",
        parent=mem_node,
        critical=True
    )

    leaf_ram = evaluator.add_leaf(
        id=f"{model_node_id}_ram_is_8gb",
        desc="RAM capacity equals 8 GB",
        parent=mem_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The RAM capacity is {RAM_GB} GB.",
        node=leaf_ram,
        sources=mem_urls,
        additional_instruction="Verify memory capacity on Apple's official technical specifications."
    )


async def verify_storage(
    evaluator: Evaluator,
    parent_node,
    model_node_id: str,
    model: Optional[ModelSpec]
) -> None:
    sto_node = evaluator.add_parallel(
        id=f"{model_node_id}_storage_options",
        desc="Storage options with official reference",
        parent=parent_node,
        critical=True
    )

    sto_urls = model.storage_urls if model else []
    evaluator.add_custom_node(
        result=has_official_apple_url(sto_urls),
        id=f"{model_node_id}_official_reference_url_for_storage",
        desc="Provides an official Apple reference URL verifying storage tiers",
        parent=sto_node,
        critical=True
    )

    leaf_tiers = evaluator.add_leaf(
        id=f"{model_node_id}_storage_includes_128_256_512_1tb",
        desc="Storage options include 128GB, 256GB, 512GB, and 1TB tiers",
        parent=sto_node,
        critical=True
    )
    claim_tiers = "Available storage options include 128GB, 256GB, 512GB, and 1TB."
    await evaluator.verify(
        claim=claim_tiers,
        node=leaf_tiers,
        sources=sto_urls,
        additional_instruction="Verify that all four tiers (128GB, 256GB, 512GB, 1TB) are offered; accept '1 TB' formatting."
    )


async def verify_display(
    evaluator: Evaluator,
    parent_node,
    model_node_id: str,
    model: Optional[ModelSpec],
    expected_tech: str,
    expected_res_str: str,
    expected_brightness_nits: int
) -> None:
    dsp_node = evaluator.add_parallel(
        id=f"{model_node_id}_display_specifications",
        desc=f"Display specifications with official reference",
        parent=parent_node,
        critical=True
    )

    dsp_urls = model.display_urls if model else []
    evaluator.add_custom_node(
        result=has_official_apple_url(dsp_urls),
        id=f"{model_node_id}_official_reference_url_for_display",
        desc="Provides an official Apple reference URL verifying display specs",
        parent=dsp_node,
        critical=True
    )

    # Display technology
    leaf_tech = evaluator.add_leaf(
        id=f"{model_node_id}_display_technology_is_liquid_retina",
        desc="Display technology type is Liquid Retina",
        parent=dsp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The display technology is {expected_tech}.",
        node=leaf_tech,
        sources=dsp_urls,
        additional_instruction="Confirm 'Liquid Retina' is specified for this model variant."
    )

    # Resolution
    leaf_res = evaluator.add_leaf(
        id=f"{model_node_id}_resolution_is_{expected_res_str.replace('×', 'x').replace('X', 'x')}",
        desc=f"Display resolution equals {expected_res_str} pixels",
        parent=dsp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The display resolution is {expected_res_str} pixels.",
        node=leaf_res,
        sources=dsp_urls,
        additional_instruction="Allow minor formatting variations (e.g., 'x' vs '×')."
    )

    # Brightness
    leaf_bri = evaluator.add_leaf(
        id=f"{model_node_id}_brightness_is_{expected_brightness_nits}_nits",
        desc=f"Maximum brightness equals {expected_brightness_nits} nits",
        parent=dsp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The maximum display brightness is {expected_brightness_nits} nits.",
        node=leaf_bri,
        sources=dsp_urls,
        additional_instruction="Verify the maximum SDR brightness value on Apple's tech specs page."
    )


async def verify_connectivity(
    evaluator: Evaluator,
    parent_node,
    model_node_id: str,
    model: Optional[ModelSpec]
) -> None:
    conn_node = evaluator.add_parallel(
        id=f"{model_node_id}_connectivity_standards",
        desc="Connectivity standards with official reference",
        parent=parent_node,
        critical=True
    )

    conn_urls = model.connectivity_urls if model else []
    evaluator.add_custom_node(
        result=has_official_apple_url(conn_urls),
        id=f"{model_node_id}_official_reference_url_for_connectivity",
        desc="Provides an official Apple reference URL verifying Wi‑Fi and Bluetooth specs",
        parent=conn_node,
        critical=True
    )

    # Wi‑Fi 6E (802.11ax)
    leaf_wifi = evaluator.add_leaf(
        id=f"{model_node_id}_wifi_is_6e_80211ax",
        desc="Wi‑Fi standard is Wi‑Fi 6E (802.11ax)",
        parent=conn_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Wi‑Fi standard is {WIFI_STD}.",
        node=leaf_wifi,
        sources=conn_urls,
        additional_instruction="Confirm the presence of Wi‑Fi 6E (802.11ax) support on Apple's page."
    )

    # Bluetooth 5.3
    leaf_bt = evaluator.add_leaf(
        id=f"{model_node_id}_bluetooth_is_5_3",
        desc="Bluetooth version is 5.3",
        parent=conn_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Bluetooth version is {BT_VERSION}.",
        node=leaf_bt,
        sources=conn_urls,
        additional_instruction="Confirm Bluetooth 5.3 support on Apple's page."
    )


async def verify_battery(
    evaluator: Evaluator,
    parent_node,
    model_node_id: str,
    model: Optional[ModelSpec],
    expected_battery_wh: float
) -> None:
    bat_node = evaluator.add_parallel(
        id=f"{model_node_id}_battery_capacity",
        desc="Battery capacity with official reference",
        parent=parent_node,
        critical=True
    )

    bat_urls = model.battery_urls if model else []
    evaluator.add_custom_node(
        result=has_official_apple_url(bat_urls),
        id=f"{model_node_id}_official_reference_url_for_battery",
        desc="Provides an official Apple reference URL verifying battery capacity",
        parent=bat_node,
        critical=True
    )

    leaf_bat = evaluator.add_leaf(
        id=f"{model_node_id}_battery_capacity_is_{str(expected_battery_wh).replace('.', '_')}_wh",
        desc=f"Battery capacity equals {expected_battery_wh} Wh",
        parent=bat_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The battery capacity is {expected_battery_wh} Wh.",
        node=leaf_bat,
        sources=bat_urls,
        additional_instruction="Look for battery capacity specified in watt‑hours on Apple's official tech specs page."
    )


async def verify_model_variant(
    evaluator: Evaluator,
    root_node,
    model_node_id: str,
    model_desc: str,
    extracted_models: TabletModelsExtraction,
    target_size_inch: float,
    display_tech: str,
    display_res_str: str,
    display_brightness_nits: int,
    battery_wh_expected: float
) -> None:
    model_parent = evaluator.add_parallel(
        id=model_node_id,
        desc=model_desc,
        parent=root_node,
        critical=False
    )

    model = pick_model_for_size(extracted_models.models, target_size_inch)

    # 1) Identification & Eligibility
    await verify_identification_and_eligibility(
        evaluator,
        model_parent,
        model_node_id,
        model_desc,
        model,
        expected_screen_size_inch=target_size_inch
    )

    # 2) Processor Architecture
    await verify_processor_architecture(
        evaluator,
        model_parent,
        model_node_id,
        model
    )

    # 3) Memory Configuration
    await verify_memory(
        evaluator,
        model_parent,
        model_node_id,
        model
    )

    # 4) Storage Options
    await verify_storage(
        evaluator,
        model_parent,
        model_node_id,
        model
    )

    # 5) Display Specifications
    await verify_display(
        evaluator,
        model_parent,
        model_node_id,
        model,
        expected_tech=display_tech,
        expected_res_str=display_res_str,
        expected_brightness_nits=display_brightness_nits
    )

    # 6) Connectivity Standards
    await verify_connectivity(
        evaluator,
        model_parent,
        model_node_id,
        model
    )

    # 7) Battery Capacity
    await verify_battery(
        evaluator,
        model_parent,
        model_node_id,
        model,
        expected_battery_wh=battery_wh_expected
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
    Evaluate an answer for the Apple tablets March 2025 task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root should be non-critical to allow aggregation across models
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

    # Extract models and their specs + official references
    extracted = await evaluator.extract(
        prompt=prompt_extract_models(),
        template_class=TabletModelsExtraction,
        extraction_name="tablet_models_march_2025_m_series_3nm"
    )

    # Record ground-truth constraints (for context in summary)
    evaluator.add_ground_truth({
        "announcement_date_expected": ANNOUNCEMENT_DATE_STR,
        "release_date_expected": RELEASE_DATE_STR,
        "processor_process_expected": PROCESS_NODE,
        "processor_architecture_expected": {
            "cpu_total": CPU_TOTAL,
            "cpu_performance": CPU_PERF,
            "cpu_efficiency": CPU_EFF,
            "gpu_cores": GPU_CORES,
            "neural_engine_cores": NE_CORES,
            "hardware_ray_tracing_supported": RAY_TRACING_SUPPORTED
        },
        "memory_expected_gb": RAM_GB,
        "storage_expected": STORAGE_TIERS,
        "display_expected": {
            "11_inch": {"tech": DISPLAY_11_TECH, "resolution": DISPLAY_11_RES, "brightness_nits": DISPLAY_11_BRIGHTNESS},
            "13_inch": {"tech": DISPLAY_13_TECH, "resolution": DISPLAY_13_RES, "brightness_nits": DISPLAY_13_BRIGHTNESS}
        },
        "connectivity_expected": {"wifi": WIFI_STD, "bluetooth": BT_VERSION},
        "battery_expected": {"11_inch_wh": BATTERY_WH_11, "13_inch_wh": BATTERY_WH_13}
    })

    # Verify two qualifying models: 11-inch and 13-inch variants
    await verify_model_variant(
        evaluator=evaluator,
        root_node=root,
        model_node_id="model_1_11_inch",
        model_desc="Qualifying tablet model: 11-inch variant",
        extracted_models=extracted,
        target_size_inch=11.0,
        display_tech=DISPLAY_11_TECH,
        display_res_str=DISPLAY_11_RES,
        display_brightness_nits=DISPLAY_11_BRIGHTNESS,
        battery_wh_expected=BATTERY_WH_11
    )

    await verify_model_variant(
        evaluator=evaluator,
        root_node=root,
        model_node_id="model_2_13_inch",
        model_desc="Qualifying tablet model: 13-inch variant",
        extracted_models=extracted,
        target_size_inch=13.0,
        display_tech=DISPLAY_13_TECH,
        display_res_str=DISPLAY_13_RES,
        display_brightness_nits=DISPLAY_13_BRIGHTNESS,
        battery_wh_expected=BATTERY_WH_13
    )

    # Return structured result
    return evaluator.get_summary()