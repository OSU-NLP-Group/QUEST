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
TASK_ID = "gaming_laptop_spec_check"
TASK_DESCRIPTION = """
I am looking for a high-performance gaming laptop suitable for both gaming and content creation work. Please identify one gaming laptop model currently available for purchase that meets ALL of the following specifications:

1. At least 16GB of RAM
2. Display with at least 144Hz refresh rate
3. USB-C port supporting Power Delivery with at least 60W charging capability
4. Discrete GPU with at least 8GB of dedicated VRAM
5. Display resolution of at least 1920x1080 (Full HD)
6. CPU with at least 6 cores
7. Display size between 15 and 17 inches (inclusive)
8. At least 512GB of SSD storage
9. Wi-Fi 6 (802.11ax) or newer wireless standard support
10. At least 3 USB ports total (any combination of USB-A and USB-C)
11. Display response time of 5ms or less
12. Weight of 2.5kg or less
13. RGB backlighting on the keyboard with customizable colors
14. Battery life of at least 4 hours under typical usage
15. At least one Thunderbolt 4 or USB4 port
16. Display covering at least 95% of the sRGB color space

For the laptop you identify, please provide:
- The exact model name and manufacturer
- A direct link to the manufacturer's official product page or specification sheet
- A direct link to at least one online retailer where the laptop is currently available for purchase

Ensure all specifications are verifiable from the official product documentation.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LaptopExtraction(BaseModel):
    # Identity and links
    manufacturer: Optional[str] = None
    model_name: Optional[str] = None
    official_product_url: Optional[str] = None
    retailer_urls: List[str] = Field(default_factory=list)

    # Specs as strings to maximize compatibility with varied answer formats
    ram: Optional[str] = None
    refresh_rate_hz: Optional[str] = None
    usb_c_pd_watts: Optional[str] = None
    discrete_gpu_model: Optional[str] = None
    gpu_vram: Optional[str] = None
    display_resolution: Optional[str] = None
    cpu_model: Optional[str] = None
    cpu_core_count: Optional[str] = None
    display_size_inches: Optional[str] = None
    storage_capacity: Optional[str] = None
    wifi_standard: Optional[str] = None
    usb_port_count_total: Optional[str] = None
    response_time_ms: Optional[str] = None
    weight_kg: Optional[str] = None
    keyboard_rgb_customizable: Optional[str] = None
    battery_life_hours: Optional[str] = None
    high_speed_port_type: Optional[str] = None  # e.g., "Thunderbolt 4", "USB4"
    srgb_coverage_percent: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_laptop_info() -> str:
    return """
Extract exactly one gaming laptop (the FIRST one if multiple are mentioned) from the answer.

Return a JSON object with these fields (use strings where applicable; do NOT invent values):
- manufacturer: The brand/manufacturer name.
- model_name: The exact model name or SKU (e.g., "ROG Strix G16 G614JZ").
- official_product_url: A direct link (URL) to the official manufacturer product page or specification sheet for the exact model/SKU.
- retailer_urls: A list of direct links to at least one retailer product page (can include more; keep as many as are present).

Also extract the following specs as they appear in the answer (strings are fine; if absent, set to null):
- ram
- refresh_rate_hz
- usb_c_pd_watts
- discrete_gpu_model
- gpu_vram
- display_resolution
- cpu_model
- cpu_core_count
- display_size_inches
- storage_capacity
- wifi_standard
- usb_port_count_total
- response_time_ms
- weight_kg
- keyboard_rgb_customizable
- battery_life_hours
- high_speed_port_type
- srgb_coverage_percent

Rules:
- Only use what is explicitly present in the answer text.
- If multiple models are mentioned, extract the first fully described model.
- Preserve units and qualifiers (e.g., "16GB", "165Hz", "100W PD", "2.4 kg", "1TB", "Wi‑Fi 6E", "3ms GtG").
- If URLs are in markdown format, extract the actual link.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_text(v: Optional[str]) -> bool:
    return v is not None and str(v).strip() != ""


def _first_or_none(urls: List[str]) -> Optional[str]:
    return urls[0] if urls else None


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, root, info: LaptopExtraction) -> None:
    """
    Build the verification tree and run verifications based on the rubric.
    The "Gaming_Laptop" node is critical and aggregates all spec checks in parallel.
    """

    # 0) Submission prerequisites under root (critical gates)
    prereq_model = evaluator.add_custom_node(
        result=_has_text(info.manufacturer) and _has_text(info.model_name),
        id="Model_Manufacturer_Provided",
        desc="Exact model name and manufacturer are provided",
        parent=root,
        critical=True
    )

    prereq_official = evaluator.add_custom_node(
        result=_has_text(info.official_product_url),
        id="Official_URL_Provided",
        desc="Official product page or specification sheet URL is provided",
        parent=root,
        critical=True
    )

    prereq_retailer = evaluator.add_custom_node(
        result=bool(info.retailer_urls),
        id="Retailer_URL_Provided",
        desc="At least one retailer URL is provided",
        parent=root,
        critical=True
    )

    # Optional: verify official URL is indeed an official manufacturer product/spec page
    official_url_verify_node = evaluator.add_leaf(
        id="Official_URL_Is_Manufacturer",
        desc="Official URL is the manufacturer's official product/spec page for the identified model",
        parent=root,
        critical=True
    )
    official_claim = (
        f"This URL is the official manufacturer product page or official specification document for the "
        f"{(info.manufacturer or '').strip()} {(info.model_name or '').strip()}."
    )
    await evaluator.verify(
        claim=official_claim,
        node=official_url_verify_node,
        sources=info.official_product_url,
        additional_instruction="Confirm the page is hosted by the laptop's manufacturer and corresponds to the exact model/SKU."
    )

    # 1) Gaming_Laptop node (critical, parallel) as per rubric
    gaming_node = evaluator.add_parallel(
        id="Gaming_Laptop",
        desc="Verify that the proposed gaming laptop meets all specified technical requirements",
        parent=root,
        critical=True
    )

    official_src = info.official_product_url if _has_text(info.official_product_url) else None

    # Helper to add a spec verification leaf quickly
    async def add_spec_leaf(node_id: str, desc: str, claim: str, add_ins: str) -> None:
        spec_node = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=gaming_node,
            critical=True
        )
        await evaluator.verify(
            claim=claim,
            node=spec_node,
            sources=official_src,
            additional_instruction=add_ins
        )

    # Specifications (all critical) — claims grounded on the official product URL
    manu_model = f"{(info.manufacturer or '').strip()} {(info.model_name or '').strip()}".strip()

    # 1. RAM ≥ 16GB
    await add_spec_leaf(
        "RAM_Specification",
        "The laptop must have at least 16GB of RAM",
        f"The official page for {manu_model} indicates the configuration has at least 16 GB of system RAM.",
        "Accept 16GB, 32GB, 64GB, etc. If multiple configurations exist, this is satisfied only if a 16GB-or-higher configuration "
        "is explicitly listed for this model/SKU (not just 'maximum capacity')."
    )

    # 2. Refresh rate ≥ 144Hz
    await add_spec_leaf(
        "Display_Refresh_Rate",
        "The laptop display must have a refresh rate of at least 144Hz",
        f"The official page for {manu_model} states the built-in display refresh rate is at least 144 Hz.",
        "Accept 144Hz, 150Hz, 165Hz, 240Hz, 360Hz, etc. If multiple panel options exist, at least one SKU for this model must be ≥144Hz."
    )

    # 3. USB-C PD ≥ 60W
    await add_spec_leaf(
        "USB_C_Power_Delivery",
        "The laptop must have at least one USB-C port supporting Power Delivery with at least 60W charging capability",
        f"The official page for {manu_model} indicates at least one USB-C port supports Power Delivery charging of 60W or higher.",
        "Look for 'USB-C PD', 'Power Delivery', 'Type-C charging', 'PD 3.0', 'up to 65W/90W/100W'."
    )

    # 4. Discrete GPU VRAM ≥ 8GB
    await add_spec_leaf(
        "GPU_VRAM",
        "The laptop's discrete GPU must have at least 8GB of dedicated VRAM",
        f"The official page for {manu_model} shows the discrete GPU in the specified configuration has at least 8 GB of dedicated VRAM.",
        "Accept expressions like '8GB GDDR6', '12GB GDDR6', etc. Ensure it's discrete GPU VRAM (not shared system memory)."
    )

    # 5. Resolution ≥ 1920x1080
    await add_spec_leaf(
        "Display_Resolution",
        "The laptop display must have a resolution of at least 1920x1080 (Full HD)",
        f"The official page for {manu_model} indicates the display resolution is at least 1920×1080.",
        "Accept FHD (1920×1080) or higher (e.g., 2560×1440, 3840×2160, QHD, 4K UHD)."
    )

    # 6. CPU cores ≥ 6
    await add_spec_leaf(
        "CPU_Cores",
        "The laptop's CPU must have at least 6 cores",
        f"The official page for {manu_model} indicates the CPU has 6 or more cores.",
        "Accept wording like '6-core', '8-core', '12-core', or core-counts in CPU model specs (e.g., Intel Core i7-12700H with 14 cores)."
    )

    # 7. Display size between 15 and 17 inches inclusive
    await add_spec_leaf(
        "Display_Size",
        "The laptop display size must be between 15 and 17 inches (inclusive)",
        f"The official page for {manu_model} indicates the screen size is within 15–17 inches inclusive.",
        "Accept sizes like 15.0\", 15.6\", 16.0\", 16.1\", 17.0\", 17.3\". Inclusive range check."
    )

    # 8. Storage ≥ 512GB SSD
    await add_spec_leaf(
        "Storage_Capacity",
        "The laptop must have at least 512GB of SSD storage",
        f"The official page for {manu_model} indicates the configuration includes at least 512 GB of SSD storage.",
        "Accept 512GB, 1TB, 2TB, etc. If options vary, at least one documented configuration for this model must be ≥512GB SSD."
    )

    # 9. Wi‑Fi 6 or newer
    await add_spec_leaf(
        "WiFi_Standard",
        "The laptop must support Wi-Fi 6 (802.11ax) or newer wireless standard",
        f"The official page for {manu_model} indicates Wi‑Fi 6 (802.11ax) or newer (e.g., Wi‑Fi 6E or Wi‑Fi 7).",
        "Accept 'Wi‑Fi 6', '802.11ax', 'Wi‑Fi 6E', 'Wi‑Fi 7'."
    )

    # 10. USB port count ≥ 3 (any combination)
    await add_spec_leaf(
        "USB_Port_Count",
        "The laptop must have at least 3 USB ports total (any combination of USB-A and USB-C)",
        f"The official page for {manu_model} indicates a total of three or more USB ports (sum of USB-A and USB-C).",
        "Count all USB-A and USB-C ports listed. Ignore HDMI/DisplayPort not labeled as USB."
    )

    # 11. Response time ≤ 5ms
    await add_spec_leaf(
        "Display_Response_Time",
        "The laptop display must have a response time of 5ms or less",
        f"The official page for {manu_model} indicates the panel response time is 5 ms or faster.",
        "Accept terms like '5ms', '3ms', '1ms', 'GtG 3ms'."
    )

    # 12. Weight ≤ 2.5 kg
    await add_spec_leaf(
        "Weight_Limit",
        "The laptop must weigh 2.5kg or less",
        f"The official page for {manu_model} indicates the weight is 2.5 kg or less.",
        "Accept approximate or range values if the maximum is ≤ 2.5 kg. If weight varies by configuration, at least one is ≤ 2.5 kg."
    )

    # 13. Keyboard RGB customizable
    await add_spec_leaf(
        "Keyboard_RGB",
        "The laptop must have RGB backlighting on the keyboard with customizable colors",
        f"The official page for {manu_model} states the keyboard has customizable RGB backlighting.",
        "Accept 'per-key RGB', '4-zone RGB', or mention of software allowing color customization."
    )

    # 14. Battery life ≥ 4 hours
    await add_spec_leaf(
        "Battery_Life",
        "The laptop must provide at least 4 hours of battery life under typical usage",
        f"The official page for {manu_model} indicates typical battery life is at least 4 hours.",
        "Accept statements like 'up to 6 hours', 'up to 8 hours'. Ensure it's typical usage, not standby."
    )

    # 15. At least one Thunderbolt 4 or USB4 port
    await add_spec_leaf(
        "High_Speed_Port",
        "The laptop must have at least one Thunderbolt 4 or USB4 port",
        f"The official page for {manu_model} indicates at least one port is Thunderbolt 4 or USB4.",
        "Accept 'Thunderbolt 4', 'USB4'. If multiple variants, at least one configuration must list TB4/USB4."
    )

    # 16. ≥ 95% sRGB coverage
    await add_spec_leaf(
        "Color_Gamut",
        "The laptop display must cover at least 95% of the sRGB color space",
        f"The official page for {manu_model} indicates the display covers 95% or more of the sRGB color space.",
        "Accept 'sRGB 95%', 'sRGB 100%', or equivalent phrasing. If multiple panels exist, at least one must meet this."
    )

    # Optional, non-critical verification that a retailer page corresponds to the same model
    # Placed under root (non-critical) to avoid failing otherwise-correct answers due to dynamic availability
    retailer_url = _first_or_none(info.retailer_urls)
    if retailer_url:
        retailer_match_node = evaluator.add_leaf(
            id="Retailer_Model_Match",
            desc="Retailer product page corresponds to the same model",
            parent=root,
            critical=False
        )
        await evaluator.verify(
            claim=f"The retailer page refers to the same {manu_model} laptop model.",
            node=retailer_match_node,
            sources=[retailer_url],
            additional_instruction="Check that the retailer page lists the same manufacturer and model name/SKU."
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
    Evaluate an answer for the gaming laptop specification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
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

    # Extract laptop information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_laptop_info(),
        template_class=LaptopExtraction,
        extraction_name="laptop_extraction"
    )

    # Record some custom info to aid debugging
    evaluator.add_custom_info(
        info={
            "extracted_model": extracted.model_name,
            "extracted_manufacturer": extracted.manufacturer,
            "official_url": extracted.official_product_url,
            "retailer_urls": extracted.retailer_urls[:3],
        },
        info_type="extraction_debug",
        info_name="extracted_overview"
    )

    # Build tree and verify according to rubric
    await build_and_verify_tree(evaluator, root, extracted)

    # Return structured result
    return evaluator.get_summary()