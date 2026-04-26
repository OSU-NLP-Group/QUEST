import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ipad_air_11_m2_specs_2024"
TASK_DESCRIPTION = (
    "A technology journalist is preparing a comprehensive technical specifications database for iPad Air models. "
    "For the iPad Air with an 11-inch display powered by the Apple M2 chip that was released in May 2024, provide complete technical specifications including: "
    "(1) Processor specifications: the specific chip model, complete CPU core configuration (including the number of performance and efficiency cores), GPU core count, and RAM capacity. "
    "(2) Display specifications: the screen size, display technology type, and refresh rate. "
    "(3) Camera specifications: the rear camera resolution and aperture, and the front camera resolution, aperture, and field of view angle. "
    "(4) Storage specifications: all available storage configuration options. "
    "(5) Connectivity specifications: the port type, data transfer speed standard and maximum speed, and maximum supported charging power. "
    "(6) Accessory compatibility: which Apple Pencil model is supported. "
    "(7) Release information: the month and year of release. "
    "For each specification category, provide the exact technical details as officially specified by Apple."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProcessorSpecs(BaseModel):
    chip_model: Optional[str] = None  # e.g., "Apple M2"
    cpu_total_cores: Optional[str] = None  # e.g., "8-core"
    cpu_performance_cores: Optional[str] = None  # e.g., "4 performance cores"
    cpu_efficiency_cores: Optional[str] = None  # e.g., "4 efficiency cores"
    gpu_core_count: Optional[str] = None  # e.g., "9-core GPU"
    memory_ram: Optional[str] = None  # e.g., "8GB"
    sources: List[str] = Field(default_factory=list)


class DisplaySpecs(BaseModel):
    screen_size: Optional[str] = None  # e.g., "11-inch"
    display_tech: Optional[str] = None  # e.g., "Liquid Retina IPS LCD"
    refresh_rate: Optional[str] = None  # e.g., "60Hz"
    sources: List[str] = Field(default_factory=list)


class CameraSpecs(BaseModel):
    rear_camera_resolution: Optional[str] = None  # e.g., "12MP Wide"
    rear_camera_aperture: Optional[str] = None  # e.g., "f/1.8"
    front_camera_resolution: Optional[str] = None  # e.g., "12MP"
    front_camera_aperture: Optional[str] = None  # e.g., "f/2.4"
    front_camera_fov: Optional[str] = None  # e.g., "122°"
    sources: List[str] = Field(default_factory=list)


class StorageSpecs(BaseModel):
    storage_options: List[str] = Field(default_factory=list)  # e.g., ["128GB", "256GB", "512GB", "1TB"]
    sources: List[str] = Field(default_factory=list)


class ConnectivitySpecs(BaseModel):
    port_type: Optional[str] = None  # e.g., "USB-C"
    data_transfer_standard: Optional[str] = None  # e.g., "USB 3.1 Gen 2" or "USB 2.0"
    max_data_speed: Optional[str] = None  # e.g., "10Gbps" or "480Mbps"
    max_charging_power: Optional[str] = None  # e.g., "30W" or "45W"
    sources: List[str] = Field(default_factory=list)


class AccessorySpecs(BaseModel):
    apple_pencil_model: Optional[str] = None  # e.g., "Apple Pencil Pro"
    sources: List[str] = Field(default_factory=list)


class ReleaseInfo(BaseModel):
    release_month: Optional[str] = None  # e.g., "May"
    release_year: Optional[str] = None  # e.g., "2024"
    sources: List[str] = Field(default_factory=list)


class IpadAirSpecs(BaseModel):
    processor: ProcessorSpecs = Field(default_factory=ProcessorSpecs)
    display: DisplaySpecs = Field(default_factory=DisplaySpecs)
    camera: CameraSpecs = Field(default_factory=CameraSpecs)
    storage: StorageSpecs = Field(default_factory=StorageSpecs)
    connectivity: ConnectivitySpecs = Field(default_factory=ConnectivitySpecs)
    accessory: AccessorySpecs = Field(default_factory=AccessorySpecs)
    release: ReleaseInfo = Field(default_factory=ReleaseInfo)
    general_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_specs() -> str:
    return """
    Extract detailed technical specifications explicitly stated in the answer for the iPad Air (11‑inch) with Apple M2 chip (released May 2024). 
    Only extract information that the answer explicitly provides. Do NOT invent missing values.
    For each category, also extract all URLs explicitly cited in the answer that support the specs for that category. 
    Prefer official Apple URLs (apple.com) when available; include them exactly as shown. If a category has no URL cited, return an empty list for its sources.

    Return JSON with this structure:

    {
      "processor": {
        "chip_model": string|null,
        "cpu_total_cores": string|null,                     // e.g., "8-core"
        "cpu_performance_cores": string|null,               // e.g., "4 performance cores"
        "cpu_efficiency_cores": string|null,                // e.g., "4 efficiency cores"
        "gpu_core_count": string|null,                      // e.g., "9-core GPU"
        "memory_ram": string|null,                          // e.g., "8GB"
        "sources": string[]                                 // URLs supporting processor specs
      },
      "display": {
        "screen_size": string|null,                         // e.g., "11-inch"
        "display_tech": string|null,                        // e.g., "Liquid Retina IPS LCD"
        "refresh_rate": string|null,                        // e.g., "60Hz"
        "sources": string[]                                 // URLs supporting display specs
      },
      "camera": {
        "rear_camera_resolution": string|null,              // e.g., "12MP Wide"
        "rear_camera_aperture": string|null,                // e.g., "f/1.8"
        "front_camera_resolution": string|null,             // e.g., "12MP"
        "front_camera_aperture": string|null,               // e.g., "f/2.4"
        "front_camera_fov": string|null,                    // e.g., "122°"
        "sources": string[]                                 // URLs supporting camera specs
      },
      "storage": {
        "storage_options": string[],                        // e.g., ["128GB","256GB","512GB","1TB"]
        "sources": string[]                                 // URLs supporting storage options
      },
      "connectivity": {
        "port_type": string|null,                           // e.g., "USB-C"
        "data_transfer_standard": string|null,              // e.g., "USB 3.1 Gen 2" or "USB 2.0"
        "max_data_speed": string|null,                      // e.g., "10Gbps" or "480Mbps"
        "max_charging_power": string|null,                  // e.g., "30W" or "45W"
        "sources": string[]                                 // URLs supporting connectivity specs
      },
      "accessory": {
        "apple_pencil_model": string|null,                  // e.g., "Apple Pencil Pro"
        "sources": string[]                                 // URLs supporting accessory compatibility
      },
      "release": {
        "release_month": string|null,                       // e.g., "May"
        "release_year": string|null,                        // e.g., "2024"
        "sources": string[]                                 // URLs supporting release information
      },
      "general_sources": string[]                           // Any general Apple URLs cited that broadly cover multiple specs
    }

    Rules:
    - Extract only what the answer explicitly includes; otherwise return null or empty lists.
    - Keep values as strings (e.g., "8GB", "60Hz", "f/1.8", "12MP").
    - Extract URLs exactly. Include full valid URLs; accept plain, markdown, or embedded formats.
    - Focus strictly on the 11-inch iPad Air (M2) released in May 2024. If the answer mixes models, extract only the 11-inch M2 Air details.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if not isinstance(x, str):
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _collect_sources(category_sources: List[str], general_sources: List[str]) -> List[str]:
    combined = (category_sources or []) + (general_sources or [])
    return _dedup_preserve_order(combined)


def _filter_apple_urls(urls: List[str]) -> List[str]:
    return [u for u in urls if isinstance(u, str) and "apple.com" in u]


def _build_additional_instruction(has_apple_urls: bool, value_missing: bool = False) -> str:
    base = (
        "Verify this claim strictly against Apple's official webpage(s) provided. "
        "Allow minor wording variations or formatting differences (e.g., '12MP' vs '12 megapixels', 'f/1.8' vs 'ƒ/1.8'). "
        "Focus on the iPad Air (11-inch, M2, released May 2024). "
    )
    if not has_apple_urls:
        base += (
            "Important: No official Apple URL is present among the provided sources for this item. "
            "Per instructions, treat the claim as NOT SUPPORTED if it cannot be verified on an Apple domain."
        )
    if value_missing:
        base += " The answer did not provide a concrete value; treat the claim as NOT SUPPORTED."
    return base


def _value_or_unknown(value: Optional[str]) -> str:
    return value if (value is not None and str(value).strip() != "") else "unknown"


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_processor_specs(evaluator: Evaluator, parent_node, specs: IpadAirSpecs) -> None:
    node = evaluator.add_parallel(
        id="ProcessorSpecifications",
        desc="Verify processor and computational specifications",
        parent=parent_node,
        critical=False
    )

    # Sources management
    proc_sources_all = _collect_sources(specs.processor.sources, specs.general_sources)
    proc_apple_sources = _filter_apple_urls(proc_sources_all)
    use_sources = proc_apple_sources if proc_apple_sources else proc_sources_all
    has_apple = len(proc_apple_sources) > 0

    # Chip Model
    chip_node = evaluator.add_leaf(
        id="ChipModel",
        desc="The device uses Apple M2 chip",
        parent=node,
        critical=True
    )
    chip_model = _value_or_unknown(specs.processor.chip_model)
    chip_claim = f"The iPad Air (11-inch, 2024) uses the chip model '{chip_model}'."
    await evaluator.verify(
        claim=chip_claim,
        node=chip_node,
        sources=use_sources,
        additional_instruction=_build_additional_instruction(has_apple, value_missing=(chip_model == "unknown"))
    )

    # CPU Configuration
    cpu_node = evaluator.add_leaf(
        id="CPUConfiguration",
        desc="The chip has 8-core CPU with 4 performance cores and 4 efficiency cores",
        parent=node,
        critical=True
    )
    cpu_total = _value_or_unknown(specs.processor.cpu_total_cores)
    cpu_perf = _value_or_unknown(specs.processor.cpu_performance_cores)
    cpu_eff = _value_or_unknown(specs.processor.cpu_efficiency_cores)
    cpu_claim = (
        f"The Apple M2 chip CPU configuration is '{cpu_total}' with '{cpu_perf}' and '{cpu_eff}'. "
        "Interpret performance vs efficiency accurately even if phrasing differs."
    )
    await evaluator.verify(
        claim=cpu_claim,
        node=cpu_node,
        sources=use_sources,
        additional_instruction=_build_additional_instruction(has_apple, value_missing=("unknown" in (cpu_total, cpu_perf, cpu_eff)))
    )

    # GPU Configuration
    gpu_node = evaluator.add_leaf(
        id="GPUConfiguration",
        desc="The chip has 9-core GPU",
        parent=node,
        critical=True
    )
    gpu_core = _value_or_unknown(specs.processor.gpu_core_count)
    gpu_claim = f"The Apple M2 chip in iPad Air has a '{gpu_core}' GPU."
    await evaluator.verify(
        claim=gpu_claim,
        node=gpu_node,
        sources=use_sources,
        additional_instruction=_build_additional_instruction(has_apple, value_missing=(gpu_core == "unknown"))
    )

    # Memory (RAM)
    mem_node = evaluator.add_leaf(
        id="Memory",
        desc="The device has 8GB RAM",
        parent=node,
        critical=True
    )
    mem_val = _value_or_unknown(specs.processor.memory_ram)
    mem_claim = f"The iPad Air (11-inch, M2) is specified with '{mem_val}' RAM."
    await evaluator.verify(
        claim=mem_claim,
        node=mem_node,
        sources=use_sources,
        additional_instruction=_build_additional_instruction(has_apple, value_missing=(mem_val == "unknown"))
    )


async def verify_display_specs(evaluator: Evaluator, parent_node, specs: IpadAirSpecs) -> None:
    node = evaluator.add_parallel(
        id="DisplaySpecifications",
        desc="Verify display characteristics",
        parent=parent_node,
        critical=False
    )

    disp_sources_all = _collect_sources(specs.display.sources, specs.general_sources)
    disp_apple_sources = _filter_apple_urls(disp_sources_all)
    use_sources = disp_apple_sources if disp_apple_sources else disp_sources_all
    has_apple = len(disp_apple_sources) > 0

    # Screen Size
    size_node = evaluator.add_leaf(
        id="ScreenSize",
        desc="The display measures 11 inches",
        parent=node,
        critical=True
    )
    size_val = _value_or_unknown(specs.display.screen_size)
    size_claim = f"The display size for iPad Air (M2) is '{size_val}'."
    await evaluator.verify(
        claim=size_claim,
        node=size_node,
        sources=use_sources,
        additional_instruction=_build_additional_instruction(has_apple, value_missing=(size_val == "unknown"))
    )

    # Display Technology
    tech_node = evaluator.add_leaf(
        id="DisplayTechnology",
        desc="The display uses Liquid Retina IPS LCD technology",
        parent=node,
        critical=True
    )
    tech_val = _value_or_unknown(specs.display.display_tech)
    tech_claim = f"The display technology is '{tech_val}' (Liquid Retina with IPS LCD)."
    await evaluator.verify(
        claim=tech_claim,
        node=tech_node,
        sources=use_sources,
        additional_instruction=_build_additional_instruction(has_apple, value_missing=(tech_val == "unknown"))
    )

    # Refresh Rate
    rr_node = evaluator.add_leaf(
        id="RefreshRate",
        desc="The display has 60Hz refresh rate",
        parent=node,
        critical=True
    )
    rr_val = _value_or_unknown(specs.display.refresh_rate)
    rr_claim = f"The display refresh rate is '{rr_val}'."
    await evaluator.verify(
        claim=rr_claim,
        node=rr_node,
        sources=use_sources,
        additional_instruction=_build_additional_instruction(has_apple, value_missing=(rr_val == "unknown"))
    )


async def verify_camera_specs(evaluator: Evaluator, parent_node, specs: IpadAirSpecs) -> None:
    node = evaluator.add_parallel(
        id="CameraSpecifications",
        desc="Verify camera system specifications",
        parent=parent_node,
        critical=False
    )

    cam_sources_all = _collect_sources(specs.camera.sources, specs.general_sources)
    cam_apple_sources = _filter_apple_urls(cam_sources_all)
    use_sources = cam_apple_sources if cam_apple_sources else cam_sources_all
    has_apple = len(cam_apple_sources) > 0

    # Rear camera resolution
    rear_res_node = evaluator.add_leaf(
        id="RearCameraResolution",
        desc="The rear camera is 12MP wide camera",
        parent=node,
        critical=True
    )
    rear_res = _value_or_unknown(specs.camera.rear_camera_resolution)
    rear_res_claim = f"The rear camera resolution is '{rear_res}'."
    await evaluator.verify(
        claim=rear_res_claim,
        node=rear_res_node,
        sources=use_sources,
        additional_instruction=_build_additional_instruction(has_apple, value_missing=(rear_res == "unknown"))
    )

    # Rear camera aperture
    rear_ap_node = evaluator.add_leaf(
        id="RearCameraAperture",
        desc="The rear camera has f/1.8 aperture",
        parent=node,
        critical=True
    )
    rear_ap = _value_or_unknown(specs.camera.rear_camera_aperture)
    rear_ap_claim = f"The rear camera aperture is '{rear_ap}'."
    await evaluator.verify(
        claim=rear_ap_claim,
        node=rear_ap_node,
        sources=use_sources,
        additional_instruction=_build_additional_instruction(has_apple, value_missing=(rear_ap == "unknown"))
    )

    # Front camera resolution
    front_res_node = evaluator.add_leaf(
        id="FrontCameraResolution",
        desc="The front camera is 12MP",
        parent=node,
        critical=True
    )
    front_res = _value_or_unknown(specs.camera.front_camera_resolution)
    front_res_claim = f"The front camera resolution is '{front_res}'."
    await evaluator.verify(
        claim=front_res_claim,
        node=front_res_node,
        sources=use_sources,
        additional_instruction=_build_additional_instruction(has_apple, value_missing=(front_res == "unknown"))
    )

    # Front camera aperture + FOV
    front_specs_node = evaluator.add_leaf(
        id="FrontCameraSpecs",
        desc="The front camera has f/2.4 aperture and 122° ultra wide angle",
        parent=node,
        critical=True
    )
    front_ap = _value_or_unknown(specs.camera.front_camera_aperture)
    front_fov = _value_or_unknown(specs.camera.front_camera_fov)
    front_specs_claim = f"The front camera aperture is '{front_ap}' and its field of view is '{front_fov}'."
    await evaluator.verify(
        claim=front_specs_claim,
        node=front_specs_node,
        sources=use_sources,
        additional_instruction=_build_additional_instruction(has_apple, value_missing=("unknown" in (front_ap, front_fov)))
    )


async def verify_storage_specs(evaluator: Evaluator, parent_node, specs: IpadAirSpecs) -> None:
    node = evaluator.add_parallel(
        id="StorageSpecifications",
        desc="Verify available storage configurations",
        parent=parent_node,
        critical=False
    )

    stor_sources_all = _collect_sources(specs.storage.sources, specs.general_sources)
    stor_apple_sources = _filter_apple_urls(stor_sources_all)
    use_sources = stor_apple_sources if stor_apple_sources else stor_sources_all
    has_apple = len(stor_apple_sources) > 0

    target_sizes = [
        ("StorageOption1", "128GB"),
        ("StorageOption2", "256GB"),
        ("StorageOption3", "512GB"),
        ("StorageOption4", "1TB"),
    ]

    for node_id, size in target_sizes:
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=f"{size} storage configuration is available",
            parent=node,
            critical=True
        )
        claim = f"The iPad Air (11-inch, M2) is available with a '{size}' storage configuration."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=use_sources,
            additional_instruction=_build_additional_instruction(has_apple, value_missing=False)
        )


async def verify_connectivity_specs(evaluator: Evaluator, parent_node, specs: IpadAirSpecs) -> None:
    node = evaluator.add_parallel(
        id="ConnectivitySpecifications",
        desc="Verify connectivity and charging specifications",
        parent=parent_node,
        critical=False
    )

    conn_sources_all = _collect_sources(specs.connectivity.sources, specs.general_sources)
    conn_apple_sources = _filter_apple_urls(conn_sources_all)
    use_sources = conn_apple_sources if conn_apple_sources else conn_sources_all
    has_apple = len(conn_apple_sources) > 0

    # Port Type
    port_node = evaluator.add_leaf(
        id="PortType",
        desc="The device has USB-C port",
        parent=node,
        critical=True
    )
    port_val = _value_or_unknown(specs.connectivity.port_type)
    port_claim = f"The iPad Air (11-inch, M2) port type is '{port_val}'."
    await evaluator.verify(
        claim=port_claim,
        node=port_node,
        sources=use_sources,
        additional_instruction=_build_additional_instruction(has_apple, value_missing=(port_val == "unknown"))
    )

    # Data Transfer Speed
    data_node = evaluator.add_leaf(
        id="DataTransferSpeed",
        desc="The USB-C port operates at USB-C 2.0 speeds of 480Mbps",
        parent=node,
        critical=True
    )
    std_val = _value_or_unknown(specs.connectivity.data_transfer_standard)
    speed_val = _value_or_unknown(specs.connectivity.max_data_speed)
    data_claim = f"The USB-C data transfer standard is '{std_val}' with a maximum speed of '{speed_val}'."
    await evaluator.verify(
        claim=data_claim,
        node=data_node,
        sources=use_sources,
        additional_instruction=_build_additional_instruction(has_apple, value_missing=("unknown" in (std_val, speed_val)))
    )

    # Charging Capability
    charge_node = evaluator.add_leaf(
        id="ChargingCapability",
        desc="The device supports charging at up to 30W or 45W",
        parent=node,
        critical=True
    )
    charge_val = _value_or_unknown(specs.connectivity.max_charging_power)
    charge_claim = f"The iPad Air (11-inch, M2) supports charging up to '{charge_val}'."
    await evaluator.verify(
        claim=charge_claim,
        node=charge_node,
        sources=use_sources,
        additional_instruction=_build_additional_instruction(has_apple, value_missing=(charge_val == "unknown"))
    )


async def verify_accessory_specs(evaluator: Evaluator, parent_node, specs: IpadAirSpecs) -> None:
    node = evaluator.add_parallel(
        id="AccessorySpecifications",
        desc="Verify accessory compatibility",
        parent=parent_node,
        critical=False
    )

    acc_sources_all = _collect_sources(specs.accessory.sources, specs.general_sources)
    acc_apple_sources = _filter_apple_urls(acc_sources_all)
    use_sources = acc_apple_sources if acc_apple_sources else acc_sources_all
    has_apple = len(acc_apple_sources) > 0

    stylus_node = evaluator.add_leaf(
        id="StylusCompatibility",
        desc="The device is compatible with Apple Pencil Pro",
        parent=node,
        critical=True
    )
    pencil_val = _value_or_unknown(specs.accessory.apple_pencil_model)
    stylus_claim = f"The iPad Air (11-inch, M2) is compatible with '{pencil_val}'."
    await evaluator.verify(
        claim=stylus_claim,
        node=stylus_node,
        sources=use_sources,
        additional_instruction=_build_additional_instruction(has_apple, value_missing=(pencil_val == "unknown"))
    )


async def verify_release_info(evaluator: Evaluator, parent_node, specs: IpadAirSpecs) -> None:
    node = evaluator.add_parallel(
        id="ReleaseInformation",
        desc="Verify release timing",
        parent=parent_node,
        critical=False
    )

    rel_sources_all = _collect_sources(specs.release.sources, specs.general_sources)
    rel_apple_sources = _filter_apple_urls(rel_sources_all)
    use_sources = rel_apple_sources if rel_apple_sources else rel_sources_all
    has_apple = len(rel_apple_sources) > 0

    release_node = evaluator.add_leaf(
        id="ReleaseMonth",
        desc="The device was released in May 2024",
        parent=node,
        critical=True
    )
    month_val = _value_or_unknown(specs.release.release_month)
    year_val = _value_or_unknown(specs.release.release_year)
    release_claim = f"The iPad Air (11-inch, M2) was released in '{month_val} {year_val}'."
    await evaluator.verify(
        claim=release_claim,
        node=release_node,
        sources=use_sources,
        additional_instruction=_build_additional_instruction(has_apple, value_missing=("unknown" in (month_val, year_val)))
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
    model: str = "o4-mini",
) -> Dict:
    """
    Build the verification tree and evaluate the answer for iPad Air (11-inch, M2) 2024 specs.
    """
    # Initialize evaluator with parallel root aggregation
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Verify all technical specifications for iPad Air 11-inch model with M2 chip released in May 2024",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured specs from the answer
    specs = await evaluator.extract(
        prompt=prompt_extract_specs(),
        template_class=IpadAirSpecs,
        extraction_name="ipad_air_11_m2_specs",
    )

    # Build verification tree based on rubric
    await verify_processor_specs(evaluator, root, specs)
    await verify_display_specs(evaluator, root, specs)
    await verify_camera_specs(evaluator, root, specs)
    await verify_storage_specs(evaluator, root, specs)
    await verify_connectivity_specs(evaluator, root, specs)
    await verify_accessory_specs(evaluator, root, specs)
    await verify_release_info(evaluator, root, specs)

    # Return evaluation summary
    return evaluator.get_summary()