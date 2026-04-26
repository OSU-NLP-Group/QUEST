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
TASK_ID = "thinkpad_x1_carbon_gen13_2025_specsheet"
TASK_DESCRIPTION = """I am considering purchasing the Lenovo ThinkPad X1 Carbon Gen 13 (2025) laptop and need to compile a comprehensive specification sheet before making my decision. Please provide the following detailed technical specifications for the base configuration model of this laptop:

1. Processor: Include the manufacturer, specific model number, generation, number of cores, and maximum clock speed in GHz
2. RAM: Include the total memory capacity in GB and the memory type (DDR4 or DDR5)
3. Storage: Include the storage capacity in GB or TB and the storage type (SSD)
4. Display Size and Resolution: Include the screen size in inches and the resolution in pixels (width × height)
5. Display Panel Type: Specify the display technology (LCD, OLED, IPS, or similar)
6. Graphics: Include the GPU manufacturer and specific model name
7. Battery: Include either the battery capacity in Wh (watt-hours) or the manufacturer-stated battery life in hours
8. Weight: Provide the laptop weight in pounds or kilograms
9. Dimensions: Include the height, width, and depth measurements in millimeters or inches
10. Operating System: Specify the exact operating system and version that comes pre-installed
11. Connectivity Ports: List the types and quantities of all available ports (USB Type-A, USB Type-C, HDMI, Thunderbolt, etc.)
12. Webcam: Specify the webcam resolution (e.g., 720p, 1080p, 2K)
13. Warranty: Specify the standard manufacturer warranty duration
14. Display Refresh Rate: Specify the display refresh rate in Hz
15. Audio/Speakers: Describe the speaker system configuration

For each specification, please provide a direct reference URL to the official Lenovo product page or an authorized retailer's specification page where this information can be verified.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class URLList(BaseModel):
    urls: List[str] = Field(default_factory=list)

class TargetModelInfo(BaseModel):
    model_name: Optional[str] = None
    base_config_indicator: Optional[str] = None  # e.g., "base configuration", "base model", "entry configuration"
    sources: List[str] = Field(default_factory=list)

class AvailabilityInfo(BaseModel):
    availability_urls: List[str] = Field(default_factory=list)

class CPUInfo(BaseModel):
    manufacturer: Optional[str] = None
    model_number: Optional[str] = None
    generation: Optional[str] = None
    cores: Optional[str] = None
    max_clock_ghz: Optional[str] = None
    sources: List[str] = Field(default_factory=list)

class RAMInfo(BaseModel):
    capacity_gb: Optional[str] = None
    memory_type: Optional[str] = None
    sources: List[str] = Field(default_factory=list)

class StorageInfo(BaseModel):
    capacity: Optional[str] = None  # e.g., "512 GB", "1 TB"
    storage_type: Optional[str] = None  # e.g., "SSD"
    sources: List[str] = Field(default_factory=list)

class DisplaySizeResolutionInfo(BaseModel):
    size_inches: Optional[str] = None
    resolution: Optional[str] = None  # e.g., "1920×1200"
    sources: List[str] = Field(default_factory=list)

class DisplayPanelInfo(BaseModel):
    panel_type: Optional[str] = None  # e.g., "IPS", "OLED", "LCD"
    sources: List[str] = Field(default_factory=list)

class GraphicsInfo(BaseModel):
    manufacturer: Optional[str] = None  # e.g., Intel, AMD, NVIDIA
    model_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)

class BatteryInfo(BaseModel):
    capacity_wh: Optional[str] = None
    life_hours: Optional[str] = None
    sources: List[str] = Field(default_factory=list)

class WeightInfo(BaseModel):
    weight: Optional[str] = None  # e.g., "1.12 kg", "2.47 lbs"
    sources: List[str] = Field(default_factory=list)

class DimensionsInfo(BaseModel):
    height: Optional[str] = None
    width: Optional[str] = None
    depth: Optional[str] = None
    sources: List[str] = Field(default_factory=list)

class OSInfo(BaseModel):
    name_version: Optional[str] = None  # e.g., "Windows 11 Pro"
    sources: List[str] = Field(default_factory=list)

class PortsInfo(BaseModel):
    ports: List[str] = Field(default_factory=list)  # e.g., ["2x Thunderbolt 4", "1x HDMI 2.1", "1x USB-A 3.2 Gen 1"]
    sources: List[str] = Field(default_factory=list)

class WebcamInfo(BaseModel):
    resolution: Optional[str] = None  # e.g., "1080p"
    sources: List[str] = Field(default_factory=list)

class WarrantyInfo(BaseModel):
    duration: Optional[str] = None  # e.g., "1-year"
    sources: List[str] = Field(default_factory=list)

class RefreshRateInfo(BaseModel):
    refresh_rate_hz: Optional[str] = None  # e.g., "60 Hz", "120 Hz"
    sources: List[str] = Field(default_factory=list)

class AudioInfo(BaseModel):
    configuration: Optional[str] = None  # e.g., "Dolby Atmos stereo speakers"
    sources: List[str] = Field(default_factory=list)

class SpecSheetExtraction(BaseModel):
    target_model: Optional[TargetModelInfo] = None
    availability: Optional[AvailabilityInfo] = None

    processor: Optional[CPUInfo] = None
    ram: Optional[RAMInfo] = None
    storage: Optional[StorageInfo] = None
    display_size_resolution: Optional[DisplaySizeResolutionInfo] = None
    display_panel: Optional[DisplayPanelInfo] = None
    graphics: Optional[GraphicsInfo] = None
    battery: Optional[BatteryInfo] = None
    weight: Optional[WeightInfo] = None
    dimensions: Optional[DimensionsInfo] = None
    operating_system: Optional[OSInfo] = None
    connectivity_ports: Optional[PortsInfo] = None
    webcam: Optional[WebcamInfo] = None
    warranty: Optional[WarrantyInfo] = None
    display_refresh_rate: Optional[RefreshRateInfo] = None
    audio_speakers: Optional[AudioInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_spec_sheet() -> str:
    return """
    Extract a structured specification sheet for Lenovo ThinkPad X1 Carbon Gen 13 (2025) from the provided answer text. 
    The user specifically asks for the base configuration model, and each specification must include a direct URL source (official Lenovo product page or an authorized retailer's spec page) cited in the answer.
    
    Return a JSON object matching the following schema. Use strings for values (even for numbers) to maximize compatibility. 
    Only extract information explicitly present in the answer text. 
    When a field is missing, set it to null; when sources are missing for a spec, return an empty array for that spec's "sources" field.
    For lists of URLs, extract actual URLs (including full protocol), not just mentions of websites.

    {
      "target_model": {
        "model_name": "<string or null>",
        "base_config_indicator": "<string or null>", 
        "sources": ["<url1>", "<url2>", ...]
      },
      "availability": {
        "availability_urls": ["<url1>", "<url2>", ...]
      },

      "processor": {
        "manufacturer": "<string or null>",
        "model_number": "<string or null>",
        "generation": "<string or null>",
        "cores": "<string or null>",
        "max_clock_ghz": "<string or null>",
        "sources": ["<url1>", "<url2>", ...]
      },
      "ram": {
        "capacity_gb": "<string or null>",
        "memory_type": "<string or null>",
        "sources": ["<url1>", "<url2>", ...]
      },
      "storage": {
        "capacity": "<string or null>",
        "storage_type": "<string or null>",
        "sources": ["<url1>", "<url2>", ...]
      },
      "display_size_resolution": {
        "size_inches": "<string or null>",
        "resolution": "<string or null>",
        "sources": ["<url1>", "<url2>", ...]
      },
      "display_panel": {
        "panel_type": "<string or null>",
        "sources": ["<url1>", "<url2>", ...]
      },
      "graphics": {
        "manufacturer": "<string or null>",
        "model_name": "<string or null>",
        "sources": ["<url1>", "<url2>", ...]
      },
      "battery": {
        "capacity_wh": "<string or null>",
        "life_hours": "<string or null>",
        "sources": ["<url1>", "<url2>", ...]
      },
      "weight": {
        "weight": "<string or null>",
        "sources": ["<url1>", "<url2>", ...]
      },
      "dimensions": {
        "height": "<string or null>",
        "width": "<string or null>",
        "depth": "<string or null>",
        "sources": ["<url1>", "<url2>", ...]
      },
      "operating_system": {
        "name_version": "<string or null>",
        "sources": ["<url1>", "<url2>", ...]
      },
      "connectivity_ports": {
        "ports": ["<port spec 1>", "<port spec 2>", "..."],
        "sources": ["<url1>", "<url2>", ...]
      },
      "webcam": {
        "resolution": "<string or null>",
        "sources": ["<url1>", "<url2>", ...]
      },
      "warranty": {
        "duration": "<string or null>",
        "sources": ["<url1>", "<url2>", ...]
      },
      "display_refresh_rate": {
        "refresh_rate_hz": "<string or null>",
        "sources": ["<url1>", "<url2>", ...]
      },
      "audio_speakers": {
        "configuration": "<string or null>",
        "sources": ["<url1>", "<url2>", ...]
      }
    }

    Important:
    - Extract only what the answer explicitly states.
    - For URL fields, return only actual URLs that appear in the answer (plain URLs or markdown links).
    - If the answer mentions a source by name (e.g., “Lenovo product page”) without a URL, do not fabricate a URL; leave the list empty.
    - Prefer extracting strings (e.g., “16 GB”, “60 Hz”, “1.12 kg”, “1920×1200”).
    - For ports, list each port type and quantity as individual list items (e.g., “2x Thunderbolt 4”, “1x HDMI 2.1”, “1x USB-A 3.2 Gen 1”).
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())

def _has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)

def _gather_all_urls(spec: SpecSheetExtraction) -> List[str]:
    urls: List[str] = []
    def extend(lst: Optional[List[str]]):
        if lst:
            for u in lst:
                if isinstance(u, str) and u.strip() and u not in urls:
                    urls.append(u)
    if spec.target_model: extend(spec.target_model.sources)
    if spec.availability: extend(spec.availability.availability_urls)
    if spec.processor: extend(spec.processor.sources)
    if spec.ram: extend(spec.ram.sources)
    if spec.storage: extend(spec.storage.sources)
    if spec.display_size_resolution: extend(spec.display_size_resolution.sources)
    if spec.display_panel: extend(spec.display_panel.sources)
    if spec.graphics: extend(spec.graphics.sources)
    if spec.battery: extend(spec.battery.sources)
    if spec.weight: extend(spec.weight.sources)
    if spec.dimensions: extend(spec.dimensions.sources)
    if spec.operating_system: extend(spec.operating_system.sources)
    if spec.connectivity_ports:
        extend(spec.connectivity_ports.sources)
    if spec.webcam: extend(spec.webcam.sources)
    if spec.warranty: extend(spec.warranty.sources)
    if spec.display_refresh_rate: extend(spec.display_refresh_rate.sources)
    if spec.audio_speakers: extend(spec.audio_speakers.sources)
    return urls

async def _add_sequential_existence_and_verification(
    evaluator: Evaluator,
    parent,
    node_id: str,
    node_desc: str,
    existence_result: bool,
    existence_desc: str,
    claim_desc: str,
    urls: Optional[List[str]],
    additional_instruction: str,
) -> None:
    """
    Create a sequential node with:
    - critical existence custom-node
    - critical verification leaf (URL-backed if URLs provided; otherwise skipped by precondition)
    """
    seq_node = evaluator.add_sequential(
        id=node_id,
        desc=node_desc,
        parent=parent,
        critical=True
    )

    # Existence check
    evaluator.add_custom_node(
        result=existence_result,
        id=f"{node_id}_exists",
        desc=existence_desc,
        parent=seq_node,
        critical=True
    )

    # Verification leaf
    verify_leaf = evaluator.add_leaf(
        id=f"{node_id}_supported",
        desc=node_desc,
        parent=seq_node,
        critical=True
    )

    await evaluator.verify(
        claim=claim_desc,
        node=verify_leaf,
        sources=(urls if urls else None),
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Verification logic builder                                                  #
# --------------------------------------------------------------------------- #
async def _build_verification_tree(evaluator: Evaluator, spec: SpecSheetExtraction) -> None:
    """
    Build the verification tree as specified by the rubric.
    The top-level "Laptop_Specifications" node is critical parallel, with each child critical.
    Most children are implemented as sequential nodes with:
      1) existence check
      2) URL-backed verification of the claim
    """
    root = evaluator.find_node("root")

    spec_root = evaluator.add_parallel(
        id="Laptop_Specifications",
        desc="Complete specification sheet for Lenovo ThinkPad X1 Carbon Gen 13 (2025) base configuration, with verifiable sources",
        parent=root,
        critical=True
    )

    all_urls = _gather_all_urls(spec)

    # 1) Target_Model_And_Base_Configuration
    tm = spec.target_model or TargetModelInfo()
    tm_exist = _nonempty(tm.model_name) and _nonempty(tm.base_config_indicator)
    tm_node = evaluator.add_sequential(
        id="Target_Model_And_Base_Configuration",
        desc="Clearly identifies the laptop as Lenovo ThinkPad X1 Carbon Gen 13 (2025) and indicates the specs correspond to the base configuration model (e.g., base SKU/variant) consistently across all listed specs",
        parent=spec_root,
        critical=True
    )
    evaluator.add_custom_node(
        result=tm_exist,
        id="Target_Model_And_Base_Configuration_exists",
        desc="Model name present and base configuration is explicitly indicated in the answer",
        parent=tm_node,
        critical=True
    )
    tm_leaf = evaluator.add_leaf(
        id="Target_Model_And_Base_Configuration_verified",
        desc="Answer text states the exact model and that specs correspond to the base configuration",
        parent=tm_node,
        critical=True
    )
    tm_claim = (
        "The answer clearly identifies the laptop as 'Lenovo ThinkPad X1 Carbon Gen 13 (2025)' "
        "and explicitly indicates that the specifications correspond to the base configuration "
        "(e.g., mentions 'base configuration', 'base model', 'entry configuration', or similar phrasing)."
    )
    await evaluator.verify(
        claim=tm_claim,
        node=tm_leaf,
        additional_instruction="Judge this by reading the provided answer text. Allow minor naming variations (e.g., 'Gen 13', '2025'). "
                              "Require that the answer asserts the base/entry configuration context."
    )

    # 2) Availability_2025
    avail = spec.availability or AvailabilityInfo()
    avail_urls = avail.availability_urls if _has_any_url(avail.availability_urls) else all_urls
    avail_exist = _has_any_url(avail.availability_urls)
    await _add_sequential_existence_and_verification(
        evaluator=evaluator,
        parent=spec_root,
        node_id="Availability_2025",
        node_desc="Uses sources indicating the model is a specific product currently available as of 2025 (e.g., official Lenovo product page or authorized retailer listing)",
        existence_result=avail_exist,
        existence_desc="At least one availability/listing URL is provided in the answer",
        claim_desc="This page is an official Lenovo product page or an authorized retailer listing for the Lenovo ThinkPad X1 Carbon Gen 13 (2025), indicating it is an actual product offering available in 2025.",
        urls=avail_urls,
        additional_instruction="Verify that the page is a real product listing for the stated model. "
                               "Authorized retailers include Lenovo, Best Buy, Amazon, B&H, Newegg, Micro Center, etc. "
                               "The page should clearly correspond to the 2025 generation (Gen 13)."
    )

    # 3) Processor_Specification_With_URL
    cpu = spec.processor or CPUInfo()
    cpu_exist = _nonempty(cpu.manufacturer) and _nonempty(cpu.model_number) and _nonempty(cpu.generation) and _nonempty(cpu.cores) and _nonempty(cpu.max_clock_ghz) and _has_any_url(cpu.sources)
    cpu_claim = (
        f"The base configuration processor for Lenovo ThinkPad X1 Carbon Gen 13 (2025) is from {cpu.manufacturer}, "
        f"model '{cpu.model_number}', generation '{cpu.generation}', with {cpu.cores} cores, "
        f"and a maximum clock speed of {cpu.max_clock_ghz}."
    )
    await _add_sequential_existence_and_verification(
        evaluator, spec_root,
        "Processor_Specification_With_URL",
        "Provides processor manufacturer, specific model number, generation, number of cores, and maximum clock speed in GHz, and includes a direct reference URL to an official Lenovo page or authorized retailer spec page verifying it",
        cpu_exist,
        "Processor fields (manufacturer, model, generation, cores, max clock) and at least one source URL are present",
        cpu_claim,
        cpu.sources,
        "Verify on the provided page that the CPU details exactly match. Allow minor formatting differences (e.g., hyphens, spacing, GHz units). "
        "If multiple CPU options exist, ensure the claimed one is listed as an option; treat it as base config if indicated or commonly the default."
    )

    # 4) RAM_Specification_With_URL
    ram = spec.ram or RAMInfo()
    ram_exist = _nonempty(ram.capacity_gb) and _nonempty(ram.memory_type) and _has_any_url(ram.sources)
    ram_claim = f"The base configuration RAM is {ram.capacity_gb} of {ram.memory_type} memory."
    await _add_sequential_existence_and_verification(
        evaluator, spec_root,
        "RAM_Specification_With_URL",
        "Provides RAM total capacity in GB and memory type (DDR4 or DDR5), and includes a direct reference URL to an official Lenovo page or authorized retailer spec page verifying it",
        ram_exist,
        "RAM capacity/type and at least one source URL are present",
        ram_claim,
        ram.sources,
        "Verify that capacity (e.g., 16 GB) and memory type (DDR4 or DDR5) are supported on the page. Allow minor formatting differences."
    )

    # 5) Storage_Specification_With_URL
    storage = spec.storage or StorageInfo()
    storage_exist = _nonempty(storage.capacity) and _nonempty(storage.storage_type) and _has_any_url(storage.sources)
    storage_claim = f"The base configuration storage is {storage.capacity} {storage.storage_type}."
    await _add_sequential_existence_and_verification(
        evaluator, spec_root,
        "Storage_Specification_With_URL",
        "Provides storage capacity (GB or TB) and storage type (SSD), and includes a direct reference URL to an official Lenovo page or authorized retailer spec page verifying it",
        storage_exist,
        "Storage capacity/type and at least one source URL are present",
        storage_claim,
        storage.sources,
        "Verify that the page shows the same capacity and storage type (e.g., SSD). Consider minor unit formatting acceptable."
    )

    # 6) Display_Size_Resolution_With_URL
    dsr = spec.display_size_resolution or DisplaySizeResolutionInfo()
    dsr_exist = _nonempty(dsr.size_inches) and _nonempty(dsr.resolution) and _has_any_url(dsr.sources)
    dsr_claim = f"The display is {dsr.size_inches} inches with a resolution of {dsr.resolution}."
    await _add_sequential_existence_and_verification(
        evaluator, spec_root,
        "Display_Size_Resolution_With_URL",
        "Provides display size in inches and resolution in pixels (width × height), and includes a direct reference URL to an official Lenovo page or authorized retailer spec page verifying it",
        dsr_exist,
        "Display size/resolution and at least one source URL are present",
        dsr_claim,
        dsr.sources,
        "Verify the size and resolution text on the page. Allow typical formatting variants (e.g., '1920 x 1200' vs '1920×1200')."
    )

    # 7) Display_Panel_Type_With_URL
    dpanel = spec.display_panel or DisplayPanelInfo()
    dpanel_exist = _nonempty(dpanel.panel_type) and _has_any_url(dpanel.sources)
    dpanel_claim = f"The display panel technology is {dpanel.panel_type}."
    await _add_sequential_existence_and_verification(
        evaluator, spec_root,
        "Display_Panel_Type_With_URL",
        "Specifies display panel technology (e.g., LCD/OLED/IPS or similar), and includes a direct reference URL to an official Lenovo page or authorized retailer spec page verifying it",
        dpanel_exist,
        "Display panel type and at least one source URL are present",
        dpanel_claim,
        dpanel.sources,
        "Verify that the page mentions the same panel technology (e.g., IPS, OLED, LCD). Synonyms like 'IPS-level' are acceptable."
    )

    # 8) Graphics_Specification_With_URL
    gfx = spec.graphics or GraphicsInfo()
    gfx_exist = _nonempty(gfx.manufacturer) and _nonempty(gfx.model_name) and _has_any_url(gfx.sources)
    gfx_claim = f"The graphics solution is from {gfx.manufacturer}, model '{gfx.model_name}'."
    await _add_sequential_existence_and_verification(
        evaluator, spec_root,
        "Graphics_Specification_With_URL",
        "Provides GPU manufacturer and specific model name, and includes a direct reference URL to an official Lenovo page or authorized retailer spec page verifying it",
        gfx_exist,
        "Graphics manufacturer/model and at least one source URL are present",
        gfx_claim,
        gfx.sources,
        "Verify integrated or discrete GPU as stated. Accept equivalent naming (e.g., 'Intel Arc integrated graphics')."
    )

    # 9) Battery_Specification_With_URL
    bat = spec.battery or BatteryInfo()
    bat_exist = (_nonempty(bat.capacity_wh) or _nonempty(bat.life_hours)) and _has_any_url(bat.sources)
    if _nonempty(bat.capacity_wh) and _nonempty(bat.life_hours):
        bat_claim = f"The battery capacity is {bat.capacity_wh}, and the manufacturer-stated battery life is {bat.life_hours}."
    elif _nonempty(bat.capacity_wh):
        bat_claim = f"The battery capacity is {bat.capacity_wh}."
    else:
        bat_claim = f"The manufacturer-stated battery life is {bat.life_hours}."
    await _add_sequential_existence_and_verification(
        evaluator, spec_root,
        "Battery_Specification_With_URL",
        "Provides either battery capacity in Wh OR manufacturer-stated battery life in hours, and includes a direct reference URL to an official Lenovo page or authorized retailer spec page verifying it",
        bat_exist,
        "Battery (capacity Wh or life hours) and at least one source URL are present",
        bat_claim,
        bat.sources,
        "Verify the capacity (Wh) or the stated battery life on the page. Minor formatting differences (e.g., '57Wh' vs '57 Wh') are acceptable."
    )

    # 10) Weight_Specification_With_URL
    wt = spec.weight or WeightInfo()
    wt_exist = _nonempty(wt.weight) and _has_any_url(wt.sources)
    wt_claim = f"The laptop weight is {wt.weight}."
    await _add_sequential_existence_and_verification(
        evaluator, spec_root,
        "Weight_Specification_With_URL",
        "Provides weight in pounds or kilograms, and includes a direct reference URL to an official Lenovo page or authorized retailer spec page verifying it",
        wt_exist,
        "Weight value and at least one source URL are present",
        wt_claim,
        wt.sources,
        "Verify the weight on the page. Accept both metric and imperial units and minor rounding differences."
    )

    # 11) Dimensions_Specification_With_URL
    dim = spec.dimensions or DimensionsInfo()
    dim_exist = _nonempty(dim.height) and _nonempty(dim.width) and _nonempty(dim.depth) and _has_any_url(dim.sources)
    dim_claim = f"The laptop dimensions are height {dim.height}, width {dim.width}, and depth {dim.depth}."
    await _add_sequential_existence_and_verification(
        evaluator, spec_root,
        "Dimensions_Specification_With_URL",
        "Provides height, width, and depth in millimeters or inches, and includes a direct reference URL to an official Lenovo page or authorized retailer spec page verifying it",
        dim_exist,
        "All three dimensions (H, W, D) and at least one source URL are present",
        dim_claim,
        dim.sources,
        "Verify all three dimensions. Accept unit differences (mm vs inches) or minor rounding."
    )

    # 12) Operating_System_With_URL
    osinfo = spec.operating_system or OSInfo()
    os_exist = _nonempty(osinfo.name_version) and _has_any_url(osinfo.sources)
    os_claim = f"The pre-installed operating system is {osinfo.name_version}."
    await _add_sequential_existence_and_verification(
        evaluator, spec_root,
        "Operating_System_With_URL",
        "Specifies the exact pre-installed operating system and version, and includes a direct reference URL to an official Lenovo page or authorized retailer spec page verifying it",
        os_exist,
        "Operating system/version and at least one source URL are present",
        os_claim,
        osinfo.sources,
        "Verify the stated OS/version in the page's specifications. Accept equivalent naming (e.g., 'Windows 11 Pro')."
    )

    # 13) Connectivity_Ports_With_URL
    ports = spec.connectivity_ports or PortsInfo()
    ports_exist = bool(ports.ports and len(ports.ports) > 0) and _has_any_url(ports.sources)
    ports_list_txt = "; ".join(ports.ports) if ports.ports else ""
    ports_claim = f"The available ports include: {ports_list_txt}."
    await _add_sequential_existence_and_verification(
        evaluator, spec_root,
        "Connectivity_Ports_With_URL",
        "Lists types and quantities of available ports (e.g., USB-A, USB-C, HDMI, Thunderbolt, etc.), and includes a direct reference URL to an official Lenovo page or authorized retailer spec page verifying it",
        ports_exist,
        "At least one port entry and at least one source URL are present",
        ports_claim,
        ports.sources,
        "Verify types and counts of ports. Accept synonyms (e.g., 'Type-C' vs 'USB-C') and that Thunderbolt 4 uses USB-C. Minor naming variants okay."
    )

    # 14) Webcam_Resolution_With_URL
    cam = spec.webcam or WebcamInfo()
    cam_exist = _nonempty(cam.resolution) and _has_any_url(cam.sources)
    cam_claim = f"The webcam resolution is {cam.resolution}."
    await _add_sequential_existence_and_verification(
        evaluator, spec_root,
        "Webcam_Resolution_With_URL",
        "Specifies webcam resolution (e.g., 720p/1080p/2K), and includes a direct reference URL to an official Lenovo page or authorized retailer spec page verifying it",
        cam_exist,
        "Webcam resolution and at least one source URL are present",
        cam_claim,
        cam.sources,
        "Verify the camera resolution on the page. Accept equivalents (e.g., 'FHD' ~ 1080p)."
    )

    # 15) Warranty_Duration_With_URL
    war = spec.warranty or WarrantyInfo()
    war_exist = _nonempty(war.duration) and _has_any_url(war.sources)
    war_claim = f"The standard manufacturer warranty is {war.duration}."
    await _add_sequential_existence_and_verification(
        evaluator, spec_root,
        "Warranty_Duration_With_URL",
        "Specifies standard manufacturer warranty duration, and includes a direct reference URL to an official Lenovo page or authorized retailer spec page verifying it",
        war_exist,
        "Warranty duration and at least one source URL are present",
        war_claim,
        war.sources,
        "Verify the standard warranty term on the page (e.g., 1-year). Region-specific nuances are acceptable if consistent with the claim."
    )

    # 16) Display_Refresh_Rate_With_URL
    rr = spec.display_refresh_rate or RefreshRateInfo()
    rr_exist = _nonempty(rr.refresh_rate_hz) and _has_any_url(rr.sources)
    rr_claim = f"The display refresh rate is {rr.refresh_rate_hz}."
    await _add_sequential_existence_and_verification(
        evaluator, spec_root,
        "Display_Refresh_Rate_With_URL",
        "Specifies display refresh rate in Hz, and includes a direct reference URL to an official Lenovo page or authorized retailer spec page verifying it",
        rr_exist,
        "Refresh rate and at least one source URL are present",
        rr_claim,
        rr.sources,
        "Verify the refresh rate on the page (e.g., 60 Hz, 120 Hz). Minor formatting differences (e.g., '60Hz') are acceptable."
    )

    # 17) Audio_Speakers_With_URL
    aud = spec.audio_speakers or AudioInfo()
    aud_exist = _nonempty(aud.configuration) and _has_any_url(aud.sources)
    aud_claim = f"The speaker/audio system configuration is: {aud.configuration}."
    await _add_sequential_existence_and_verification(
        evaluator, spec_root,
        "Audio_Speakers_With_URL",
        "Describes the speaker/audio system configuration, and includes a direct reference URL to an official Lenovo page or authorized retailer spec page verifying it",
        aud_exist,
        "Audio/speaker configuration and at least one source URL are present",
        aud_claim,
        aud.sources,
        "Verify the audio/speaker configuration (e.g., stereo speakers, Dolby Atmos) as stated on the page."
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
    Evaluate an answer for the Lenovo ThinkPad X1 Carbon Gen 13 (2025) specification sheet task.
    Builds a critical parallel rubric with sequential child checks per specification.
    """
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

    # Extraction
    spec_data = await evaluator.extract(
        prompt=prompt_extract_spec_sheet(),
        template_class=SpecSheetExtraction,
        extraction_name="spec_sheet_extraction"
    )

    # Build verification tree according to rubric
    await _build_verification_tree(evaluator, spec_data)

    return evaluator.get_summary()