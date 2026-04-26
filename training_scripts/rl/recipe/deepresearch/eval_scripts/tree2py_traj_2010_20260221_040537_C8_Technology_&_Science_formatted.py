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
TASK_ID = "gaming_laptop_2025_2026"
TASK_DESCRIPTION = """
Identify a gaming laptop model available or announced for release in 2025-2026 that meets ALL of the following technical specifications:

Display Requirements:
- Display technology: OLED or mini-LED
- Resolution: At least 2560×1440 (QHD) or higher
- Refresh rate: At least 144Hz or higher
- HDR support: VESA DisplayHDR certification (any level)

Performance Requirements:
- Processor: Intel Core Ultra 200 series or higher, AMD Ryzen 9000 series or higher, or Apple M-series
- GPU: NVIDIA RTX 5060 or higher, AMD RX 8000 series or higher, or equivalent discrete GPU
- RAM: At least 16GB
- Storage: At least 512GB NVMe SSD

Connectivity & Compatibility:
- Operating System: Windows 10 version 20H2 or later, macOS 14.1.2 or later, or compatible Linux
- USB-C: At least one USB-C port with Power Delivery support (minimum 65W)
- WiFi: WiFi 6E (802.11ax with 6GHz) or WiFi 7 (802.11be)
- Internet capability: Can support 20 Mbps or higher for cloud gaming

Additional Features:
- Battery life: At least 6 hours mixed use or 4 hours gaming
- Audio: Support for advanced Bluetooth codecs (aptX, AAC, or LDAC) AND spatial audio (Dolby Atmos, DTS:X, or equivalent)

Provide the laptop model name, manufacturer, and URL references verifying each major specification category.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DisplaySpecs(BaseModel):
    technology: Optional[str] = None  # e.g., "OLED", "Mini-LED"
    resolution: Optional[str] = None  # e.g., "2560x1600", "QHD+"
    refresh_rate: Optional[str] = None  # e.g., "144Hz", "165 Hz", "240Hz"
    hdr_certification: Optional[str] = None  # e.g., "VESA DisplayHDR 600"
    urls: List[str] = Field(default_factory=list)


class PerformanceSpecs(BaseModel):
    processor: Optional[str] = None  # e.g., "Intel Core Ultra 9 285H", "AMD Ryzen 9 9950HX", "Apple M3"
    gpu: Optional[str] = None  # e.g., "NVIDIA GeForce RTX 5070 Laptop GPU"
    ram: Optional[str] = None  # e.g., "16GB", "32 GB"
    storage: Optional[str] = None  # e.g., "1TB NVMe SSD", "512GB PCIe 4.0 NVMe SSD"
    urls: List[str] = Field(default_factory=list)


class ConnectivitySpecs(BaseModel):
    os: Optional[str] = None  # e.g., "Windows 11", "Windows 10 22H2", "Ubuntu 22.04", "macOS 15"
    usb_c_pd_watts: Optional[str] = None  # e.g., "100W USB-C PD", "65 W Power Delivery"
    wifi: Optional[str] = None  # e.g., "Wi-Fi 6E", "WiFi 7"
    internet_speed_statement: Optional[str] = None  # e.g., "Supports at least 20 Mbps internet for cloud gaming"
    urls: List[str] = Field(default_factory=list)


class AdditionalFeaturesSpecs(BaseModel):
    battery_life_mixed: Optional[str] = None  # e.g., "up to 8 hours"
    battery_life_gaming: Optional[str] = None  # e.g., "4 hours gaming"
    bluetooth_codecs: List[str] = Field(default_factory=list)  # e.g., ["aptX", "LDAC"]
    spatial_audio: List[str] = Field(default_factory=list)  # e.g., ["Dolby Atmos", "DTS:X"]
    urls: List[str] = Field(default_factory=list)


class LaptopExtraction(BaseModel):
    model_name: Optional[str] = None
    manufacturer: Optional[str] = None

    availability_year_or_window: Optional[str] = None  # e.g., "2025", "Q1 2026", "2025-2026"
    availability_urls: List[str] = Field(default_factory=list)

    display: Optional[DisplaySpecs] = None
    performance: Optional[PerformanceSpecs] = None
    connectivity: Optional[ConnectivitySpecs] = None
    additional: Optional[AdditionalFeaturesSpecs] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_laptop() -> str:
    return """
    Extract the primary gaming laptop chosen in the answer. If multiple models are mentioned, pick the first clearly recommended or first listed model. Extract the following fields exactly as stated in the answer text:

    Identification:
    - model_name: The exact model name of the laptop (e.g., "Razer Blade 16 (2025)", "ASUS ROG Zephyrus G16")
    - manufacturer: The brand/manufacturer (e.g., "Razer", "ASUS", "MSI", "Lenovo", "HP")

    Availability:
    - availability_year_or_window: The availability/release year or window (e.g., "2025", "Q1 2026", "2025-2026")
    - availability_urls: Array of URL(s) that verify the availability timeframe (include only URLs explicitly present in the answer)

    Display (Display Requirements evidence URLs should be specific to display specs for the chosen model):
    - display.technology: e.g., "OLED", "Mini-LED", "mini LED"
    - display.resolution: e.g., "2560x1600", "QHD", "QHD+", "3K", "4K", "3840x2160"
    - display.refresh_rate: e.g., "144Hz", "165 Hz", "240Hz"
    - display.hdr_certification: e.g., "VESA DisplayHDR 500", "DisplayHDR 600", or null if not mentioned
    - display.urls: Array of URL(s) that verify display specs for this model

    Performance (Performance evidence URLs should be specific to the CPU/GPU/RAM/Storage of the chosen model):
    - performance.processor: e.g., "Intel Core Ultra 9 285H", "AMD Ryzen 9 9950HX", "Apple M3"
    - performance.gpu: e.g., "NVIDIA GeForce RTX 5070 Laptop GPU", "AMD Radeon RX 8700M"
    - performance.ram: e.g., "16GB", "32 GB"
    - performance.storage: e.g., "1TB NVMe SSD", "512GB PCIe 4.0 NVMe SSD"
    - performance.urls: Array of URL(s) that verify the performance specs for this model

    Connectivity & Compatibility (URLs should be specific to ports/wireless/OS for the chosen model):
    - connectivity.os: e.g., "Windows 11", "Windows 10 22H2", "Ubuntu 22.04", "macOS 15"
    - connectivity.usb_c_pd_watts: e.g., "65W USB-C PD", "100W Power Delivery", "Thunderbolt 4 with 100W PD"
    - connectivity.wifi: e.g., "Wi-Fi 6E", "WiFi 7 (802.11be)"
    - connectivity.internet_speed_statement: verbatim statement if present (else null)
    - connectivity.urls: Array of URL(s) that verify connectivity/compatibility specs for this model

    Additional Features (URLs should be specific to battery life and audio features for this model):
    - additional.battery_life_mixed: e.g., "up to 8 hours", "6-10 hours"
    - additional.battery_life_gaming: e.g., "4 hours gaming", if present
    - additional.bluetooth_codecs: Array of codecs explicitly named (e.g., ["aptX", "AAC", "LDAC"]); if none stated, return an empty array
    - additional.spatial_audio: Array of spatial audio tech explicitly named (e.g., ["Dolby Atmos", "DTS:X", "Windows Sonic"])
    - additional.urls: Array of URL(s) that verify battery and audio features for this model

    IMPORTANT:
    - Only extract URLs that are explicitly present in the answer text (plain or markdown links). Do not invent URLs.
    - Keep all fields as strings as they appear. If a field is missing, set it to null (or an empty array for list fields).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls)


def _safe_list(urls: Optional[List[str]]) -> List[str]:
    return urls if urls is not None else []


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_laptop_identification(evaluator: Evaluator, parent, data: LaptopExtraction) -> None:
    """
    Build and verify Laptop_Identification: presence of model name and manufacturer.
    """
    ident_node = evaluator.add_parallel(
        id="Laptop_Identification",
        desc="Answer provides the laptop model name and manufacturer.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(data.model_name and data.model_name.strip()),
        id="Model_Name_Provided",
        desc="Laptop model name is stated.",
        parent=ident_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(data.manufacturer and data.manufacturer.strip()),
        id="Manufacturer_Provided",
        desc="Laptop manufacturer/brand is stated.",
        parent=ident_node,
        critical=True
    )


async def build_availability_group(evaluator: Evaluator, parent, data: LaptopExtraction) -> None:
    """
    Build availability verification group (timeframe + URLs presence).
    """
    avail_group = evaluator.add_parallel(
        id="Availability_Group",
        desc="Availability evidence for 2025–2026 timeframe.",
        parent=parent,
        critical=True
    )

    # Ensure URL evidence exists (critical)
    evaluator.add_custom_node(
        result=_has_any_url(data.availability_urls),
        id="Availability_URL_References",
        desc="One or more URL references are provided that verify the availability timeframe.",
        parent=avail_group,
        critical=True
    )

    # Verify availability timeframe 2025–2026
    avail_leaf = evaluator.add_leaf(
        id="Availability_Timeframe_2025_2026",
        desc="Laptop is available or announced for release in 2025–2026.",
        parent=avail_group,
        critical=True
    )
    model = (data.manufacturer or "").strip() + " " + (data.model_name or "").strip()
    claim = f"The {model.strip()} is available or officially announced for release in 2025 or 2026."
    await evaluator.verify(
        claim=claim,
        node=avail_leaf,
        sources=_safe_list(data.availability_urls),
        additional_instruction=(
            "Accept evidence that clearly indicates launch/announce/release/availability in calendar year 2025 or 2026 "
            "(e.g., 'Shipping in early 2025', 'Announced at CES 2026', 'Available 2025'). "
            "Reject if the URLs do not mention 2025 or 2026 for this model."
        )
    )


async def build_display_requirements(evaluator: Evaluator, parent, data: LaptopExtraction) -> None:
    disp = data.display or DisplaySpecs()
    model = (data.manufacturer or "").strip() + " " + (data.model_name or "").strip()

    node = evaluator.add_parallel(
        id="Display_Requirements",
        desc="Meets all display requirements and provides URL evidence for the display category.",
        parent=parent,
        critical=True
    )

    # URL presence for display specs
    evaluator.add_custom_node(
        result=_has_any_url(disp.urls),
        id="Display_URL_References",
        desc="One or more URL references are provided that verify the display category specs.",
        parent=node,
        critical=True
    )

    # Display technology
    tech_leaf = evaluator.add_leaf(
        id="Display_Technology",
        desc="Display technology is OLED or mini-LED.",
        parent=node,
        critical=True
    )
    claim = (
        f"The {model.strip()} uses an OLED or mini-LED display. "
        f"Stated technology: '{(disp.technology or '').strip()}'."
    )
    await evaluator.verify(
        claim=claim,
        node=tech_leaf,
        sources=_safe_list(disp.urls),
        additional_instruction="Accept if the page indicates OLED or Mini LED (mini-LED, Mini LED, miniLED). Reject IPS/LCD unless explicitly stated as mini-LED backlight."
    )

    # Resolution minimum
    res_leaf = evaluator.add_leaf(
        id="Display_Resolution_Minimum",
        desc="Display resolution is at least 2560×1440 (QHD) or higher.",
        parent=node,
        critical=True
    )
    claim = (
        f"The {model.strip()} has a display resolution of '{(disp.resolution or '').strip()}', "
        "which is at least 2560×1440 (QHD) or higher."
    )
    await evaluator.verify(
        claim=claim,
        node=res_leaf,
        sources=_safe_list(disp.urls),
        additional_instruction=(
            "Consider common labels: QHD/QHD+ (≥2560×1440), 2.5K/3K/4K where 3K/4K exceed QHD. "
            "If resolution text (e.g., 2560×1600, 3200×2000, 3840×2160) meets or exceeds QHD, accept."
        )
    )

    # Refresh rate minimum
    rr_leaf = evaluator.add_leaf(
        id="Display_Refresh_Rate_Minimum",
        desc="Refresh rate is at least 144 Hz.",
        parent=node,
        critical=True
    )
    claim = (
        f"The {model.strip()} has a display refresh rate of '{(disp.refresh_rate or '').strip()}', "
        "which is at least 144 Hz."
    )
    await evaluator.verify(
        claim=claim,
        node=rr_leaf,
        sources=_safe_list(disp.urls),
        additional_instruction="Accept if the refresh rate is ≥144Hz (e.g., 144Hz, 165Hz, 240Hz)."
    )

    # HDR VESA DisplayHDR certification
    hdr_leaf = evaluator.add_leaf(
        id="HDR_VESA_DisplayHDR",
        desc="Display has VESA DisplayHDR certification (any level).",
        parent=node,
        critical=True
    )
    claim = (
        f"The {model.strip()} display has a VESA DisplayHDR certification (any level). "
        f"Stated HDR: '{(disp.hdr_certification or '').strip()}'."
    )
    await evaluator.verify(
        claim=claim,
        node=hdr_leaf,
        sources=_safe_list(disp.urls),
        additional_instruction=(
            "Accept mentions like 'VESA DisplayHDR 400/500/600/1000', 'DisplayHDR' with any number. "
            "Reject generic 'HDR' without VESA DisplayHDR label."
        )
    )


async def build_performance_requirements(evaluator: Evaluator, parent, data: LaptopExtraction) -> None:
    perf = data.performance or PerformanceSpecs()
    model = (data.manufacturer or "").strip() + " " + (data.model_name or "").strip()

    node = evaluator.add_parallel(
        id="Performance_Requirements",
        desc="Meets all performance requirements and provides URL evidence for the performance category.",
        parent=parent,
        critical=True
    )

    # URL presence for performance specs
    evaluator.add_custom_node(
        result=_has_any_url(perf.urls),
        id="Performance_URL_References",
        desc="One or more URL references are provided that verify the performance category specs.",
        parent=node,
        critical=True
    )

    # Processor requirement
    cpu_leaf = evaluator.add_leaf(
        id="Processor_Requirement",
        desc="Processor is Intel Core Ultra 200 series or higher, AMD Ryzen 9000 series or higher, or Apple M-series.",
        parent=node,
        critical=True
    )
    claim = (
        f"The {model.strip()} uses a processor '{(perf.processor or '').strip()}' that qualifies as "
        "Intel Core Ultra 200 series or newer, AMD Ryzen 9000 series or newer, or Apple M-series."
    )
    await evaluator.verify(
        claim=claim,
        node=cpu_leaf,
        sources=_safe_list(perf.urls),
        additional_instruction=(
            "Reason based on the CPU name from the page. Examples that qualify: 'Intel Core Ultra 9 285H' (Ultra 200), "
            "'AMD Ryzen 9 9950HX' (Ryzen 9000), any Apple 'M' chip (M1/M2/M3/M4). "
            "Reject older series (e.g., Intel 13th gen, Ryzen 7000) unless explicitly an Apple M-series."
        )
    )

    # GPU requirement
    gpu_leaf = evaluator.add_leaf(
        id="GPU_Requirement",
        desc="GPU is NVIDIA RTX 5060 or higher, AMD RX 8000 series or higher, or an equivalent discrete GPU.",
        parent=node,
        critical=True
    )
    claim = (
        f"The {model.strip()} uses a discrete GPU '{(perf.gpu or '').strip()}' that qualifies as "
        "NVIDIA GeForce RTX 5060/5070/5080/5090 (Laptop GPU), or AMD Radeon RX 8000 series or higher, or equivalent."
    )
    await evaluator.verify(
        claim=claim,
        node=gpu_leaf,
        sources=_safe_list(perf.urls),
        additional_instruction=(
            "Accept NVIDIA GeForce RTX 50-series laptop GPUs (≥5060) or AMD Radeon RX 8000-series laptop GPUs. "
            "Reject older series like RTX 40 (e.g., 4060) or RX 7000 unless explicitly stated as equivalent next-gen GPU."
        )
    )

    # RAM minimum
    ram_leaf = evaluator.add_leaf(
        id="RAM_Minimum",
        desc="RAM is at least 16 GB.",
        parent=node,
        critical=True
    )
    claim = (
        f"The {model.strip()} provides RAM '{(perf.ram or '').strip()}' that is at least 16 GB."
    )
    await evaluator.verify(
        claim=claim,
        node=ram_leaf,
        sources=_safe_list(perf.urls),
        additional_instruction="Accept 16GB or more (e.g., 16GB, 32GB, 64GB). Reject 8GB."
    )

    # Storage minimum
    storage_leaf = evaluator.add_leaf(
        id="Storage_Minimum",
        desc="Storage is at least 512 GB NVMe SSD.",
        parent=node,
        critical=True
    )
    claim = (
        f"The {model.strip()} provides storage '{(perf.storage or '').strip()}' that is at least a 512GB NVMe SSD."
    )
    await evaluator.verify(
        claim=claim,
        node=storage_leaf,
        sources=_safe_list(perf.urls),
        additional_instruction=(
            "Accept 512GB or larger NVMe/PCIe SSD (e.g., 512GB/1TB/2TB). "
            "Reject HDD-only or capacities <512GB. If 'PCIe/NVMe SSD' is stated, accept."
        )
    )


async def build_connectivity_requirements(evaluator: Evaluator, parent, data: LaptopExtraction) -> None:
    conn = data.connectivity or ConnectivitySpecs()
    model = (data.manufacturer or "").strip() + " " + (data.model_name or "").strip()

    node = evaluator.add_parallel(
        id="Connectivity_Compatibility",
        desc="Meets all connectivity/compatibility requirements and provides URL evidence for this category.",
        parent=parent,
        critical=True
    )

    # URL presence
    evaluator.add_custom_node(
        result=_has_any_url(conn.urls),
        id="Connectivity_URL_References",
        desc="One or more URL references are provided that verify the connectivity & compatibility category specs.",
        parent=node,
        critical=True
    )

    # Operating system requirement
    os_leaf = evaluator.add_leaf(
        id="Operating_System_Requirement",
        desc="OS is Windows 10 v20H2 or later, macOS 14.1.2 or later, or compatible Linux.",
        parent=node,
        critical=True
    )
    claim = (
        f"The {model.strip()} runs an OS '{(conn.os or '').strip()}' that is Windows 10 20H2 or later, macOS 14.1.2 or later, "
        "or a compatible Linux distribution."
    )
    await evaluator.verify(
        claim=claim,
        node=os_leaf,
        sources=_safe_list(conn.urls),
        additional_instruction=(
            "Accept Windows 11/12 and any Windows 10 build ≥20H2. Accept macOS ≥14.1.2 if Apple device. "
            "Accept Linux if vendor states Linux support. Reject older unsupported versions."
        )
    )

    # USB-C PD requirement
    pd_leaf = evaluator.add_leaf(
        id="USB_C_PD_Requirement",
        desc="At least one USB-C port supports Power Delivery with minimum 65 W.",
        parent=node,
        critical=True
    )
    claim = (
        f"The {model.strip()} has at least one USB-C port with Power Delivery support of at least 65 W "
        f"(stated: '{(conn.usb_c_pd_watts or '').strip()}')."
    )
    await evaluator.verify(
        claim=claim,
        node=pd_leaf,
        sources=_safe_list(conn.urls),
        additional_instruction=(
            "Accept mentions like 'USB-C PD 65W/90W/100W/140W', 'USB-C charging 65W+', 'Thunderbolt 4 with 100W charging'. "
            "Reject if only data/DP without PD charging or PD < 65W."
        )
    )

    # WiFi requirement
    wifi_leaf = evaluator.add_leaf(
        id="WiFi_Requirement",
        desc="Supports WiFi 6E (802.11ax with 6GHz) or WiFi 7 (802.11be).",
        parent=node,
        critical=True
    )
    claim = (
        f"The {model.strip()} supports WiFi 6E or WiFi 7 (stated: '{(conn.wifi or '').strip()}')."
    )
    await evaluator.verify(
        claim=claim,
        node=wifi_leaf,
        sources=_safe_list(conn.urls),
        additional_instruction="Accept explicit 'Wi-Fi 6E' or 'WiFi 7/802.11be'. Reject Wi-Fi 6 (without E) unless 6GHz is stated."
    )

    # Internet capability for cloud gaming (≥20 Mbps)
    net_leaf = evaluator.add_leaf(
        id="Internet_Speed_Cloud_Gaming",
        desc="Can support 20 Mbps or higher for cloud gaming.",
        parent=node,
        critical=True
    )
    claim = (
        f"The {model.strip()} can support at least 20 Mbps internet throughput for cloud gaming."
    )
    await evaluator.verify(
        claim=claim,
        node=net_leaf,
        sources=_safe_list(conn.urls),
        additional_instruction=(
            "Reasonable inference allowed: if the laptop supports Wi‑Fi 6E/7 or Gigabit Ethernet, it clearly supports ≥20 Mbps. "
            "Accept based on wireless/ethernet capability; exact '20 Mbps' wording is not required if capability is obvious."
        )
    )


async def build_additional_features(evaluator: Evaluator, parent, data: LaptopExtraction) -> None:
    add = data.additional or AdditionalFeaturesSpecs()
    model = (data.manufacturer or "").strip() + " " + (data.model_name or "").strip()

    node = evaluator.add_parallel(
        id="Additional_Features",
        desc="Meets additional feature requirements and provides URL evidence for this category.",
        parent=parent,
        critical=True
    )

    # URL presence
    evaluator.add_custom_node(
        result=_has_any_url(add.urls),
        id="Additional_Features_URL_References",
        desc="One or more URL references are provided that verify the additional features category specs.",
        parent=node,
        critical=True
    )

    # Battery life requirement
    battery_leaf = evaluator.add_leaf(
        id="Battery_Life_Requirement",
        desc="Battery life is at least 6 hours mixed use OR at least 4 hours gaming.",
        parent=node,
        critical=True
    )
    mixed = (add.battery_life_mixed or "").strip()
    gaming = (add.battery_life_gaming or "").strip()
    claim = (
        f"The {model.strip()} provides battery life that meets the requirement: "
        f"at least 6 hours mixed use or at least 4 hours gaming. "
        f"Stated battery life: mixed='{mixed}', gaming='{gaming}'."
    )
    await evaluator.verify(
        claim=claim,
        node=battery_leaf,
        sources=_safe_list(add.urls),
        additional_instruction=(
            "Accept statements like 'up to 6 hours' or 'around 6 hours' for mixed/general use; "
            "for gaming, accept '4 hours' or more. If both are present, only one needs to meet the threshold."
        )
    )

    # Bluetooth codecs requirement
    bt_leaf = evaluator.add_leaf(
        id="Bluetooth_Codecs_Requirement",
        desc="Supports advanced Bluetooth codecs (aptX, AAC, or LDAC).",
        parent=node,
        critical=True
    )
    claim = (
        f"The {model.strip()} supports at least one advanced Bluetooth codec among aptX, AAC, or LDAC. "
        f"Stated codecs: {add.bluetooth_codecs}."
    )
    await evaluator.verify(
        claim=claim,
        node=bt_leaf,
        sources=_safe_list(add.urls),
        additional_instruction=(
            "Accept explicit support mentions for aptX (any variant), AAC, or LDAC on the model page or official docs. "
            "Reject if only generic 'Bluetooth 5.x' without codec detail."
        )
    )

    # Spatial audio requirement
    spatial_leaf = evaluator.add_leaf(
        id="Spatial_Audio_Requirement",
        desc="Supports spatial audio (Dolby Atmos, DTS:X, or equivalent).",
        parent=node,
        critical=True
    )
    claim = (
        f"The {model.strip()} supports spatial audio such as Dolby Atmos, DTS:X, or an equivalent solution. "
        f"Stated spatial audio tech: {add.spatial_audio}."
    )
    await evaluator.verify(
        claim=claim,
        node=spatial_leaf,
        sources=_safe_list(add.urls),
        additional_instruction=(
            "Accept explicit labels like 'Dolby Atmos', 'DTS:X', 'Windows Sonic', 'THX Spatial Audio', or OEM-branded spatial audio "
            "solutions that clearly indicate surround/spatial processing. Reject if no spatial audio feature is mentioned."
        )
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
    Evaluate an answer for the 2025–2026 gaming laptop selection task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Categories are independent checks under the main selection
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

    # Extract structured laptop info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_laptop(),
        template_class=LaptopExtraction,
        extraction_name="laptop_extraction"
    )

    # Build top-level critical node for the selection as per rubric
    main = evaluator.add_parallel(
        id="Gaming_Laptop_Selection",
        desc=("Verify the answer identifies one gaming laptop available/announced for 2025–2026 that satisfies all "
              "required specifications and provides URLs verifying each major specification category (Display, "
              "Performance, Connectivity & Compatibility, Additional Features)."),
        parent=root,
        critical=True
    )

    # Build subtrees
    await build_laptop_identification(evaluator, main, extracted)
    await build_availability_group(evaluator, main, extracted)
    await build_display_requirements(evaluator, main, extracted)
    await build_performance_requirements(evaluator, main, extracted)
    await build_connectivity_requirements(evaluator, main, extracted)
    await build_additional_features(evaluator, main, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()