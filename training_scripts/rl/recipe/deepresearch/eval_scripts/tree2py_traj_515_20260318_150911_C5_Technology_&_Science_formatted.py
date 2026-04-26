import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "streaming_devices_4k_under80_march2026"
TASK_DESCRIPTION = """
I'm upgrading my home entertainment system and need to find three different streaming devices that meet specific performance requirements for high-quality 4K streaming.

Each device must meet ALL of the following requirements:

Video Performance:
- Support 4K Ultra HD video output (3840 x 2160 resolution)
- Support at least one HDR format (Dolby Vision, HDR10+, HDR10, or HLG)

Hardware Specifications:
- Have at least a quad-core processor with a minimum clock speed of 1.5 GHz
- Have at least 1GB of RAM
- Provide at least 8GB of internal storage

Connectivity:
- Support Wi-Fi 5 (802.11ac) or a newer standard (such as Wi-Fi 6 or Wi-Fi 6E)
- Support Dolby Atmos or DTS audio technology

Availability and Form Factor:
- Be currently available for purchase as of March 2026 from major retailers or the manufacturer
- Have a standard retail price under $80 USD (not counting temporary sales or promotions)
- Be a streaming stick or compact streaming device (not a full-size set-top box)

For each of the three devices, please provide:
1. The complete device name and model number
2. The manufacturer
3. A reference URL from the manufacturer's website or a major retailer (such as Amazon, Best Buy, or Walmart) that confirms the device specifications
4. Verification that the device meets each of the technical requirements listed above
"""

CURRENT_MONTH_YEAR = "March 2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DeviceExtraction(BaseModel):
    device_name: Optional[str] = None
    model_number: Optional[str] = None
    manufacturer: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)

    # Claims (free-form strings from answer; used for context only)
    resolution: Optional[str] = None
    hdr_formats: List[str] = Field(default_factory=list)

    processor_description: Optional[str] = None
    ram: Optional[str] = None
    storage: Optional[str] = None

    wifi: Optional[str] = None
    audio: Optional[str] = None

    price_usd: Optional[str] = None
    availability_statement: Optional[str] = None
    form_factor: Optional[str] = None


class DevicesExtraction(BaseModel):
    devices: List[DeviceExtraction] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_devices() -> str:
    return """
Extract up to the first three streaming devices mentioned in the answer. For each device, return a JSON object with the following fields:

- device_name: The complete product name as stated
- model_number: The specific model number/code, if given (e.g., "3820R", "GJQ9T", etc.). If absent, return null.
- manufacturer: The brand/company (e.g., "Roku", "Amazon", "Google")
- reference_urls: An array of URLs explicitly cited in the answer for this device (manufacturer or major retailers such as Amazon, Best Buy, Walmart). Include all unique URLs shown for this device.

Also extract the device's claimed specs from the answer text (verbatim or close paraphrase) as context-only strings:
- resolution: e.g., "4K Ultra HD (3840 x 2160)", "2160p"
- hdr_formats: array of named HDR formats (e.g., "Dolby Vision", "HDR10", "HDR10+", "HLG") if claimed
- processor_description: e.g., "Quad-core 1.8 GHz"
- ram: e.g., "2 GB RAM"
- storage: e.g., "8 GB internal storage"
- wifi: e.g., "Wi‑Fi 5 (802.11ac)", "Wi‑Fi 6"
- audio: e.g., "Dolby Atmos", "DTS"
- price_usd: e.g., "$49.99", "MSRP $39.99"
- availability_statement: any claim about being available for purchase
- form_factor: e.g., "streaming stick", "compact dongle", "small puck"

Rules:
- Extract only what is explicitly present in the answer text.
- For URLs: Only include valid, complete URLs that appear in the answer (plain or markdown). Do not invent any URLs.
- If a field isn’t provided, set it to null (or empty list for arrays).

Return a JSON object:
{
  "devices": [
     { ... device #1 ... },
     { ... device #2 ... },
     { ... device #3 ... }
  ]
}
If fewer than 3 devices are present, return only those found.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def device_label(d: DeviceExtraction) -> str:
    parts = []
    if d.device_name:
        parts.append(d.device_name.strip())
    if d.model_number:
        parts.append(f"({d.model_number.strip()})")
    if d.manufacturer:
        parts.append(f"by {d.manufacturer.strip()}")
    return " ".join(parts) if parts else "the device"


def first_n_devices(extracted: DevicesExtraction, n: int = 3) -> List[DeviceExtraction]:
    items = list(extracted.devices[:n]) if extracted and extracted.devices else []
    while len(items) < n:
        items.append(DeviceExtraction())
    return items


def has_text(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


# --------------------------------------------------------------------------- #
# Verification logic per device                                               #
# --------------------------------------------------------------------------- #
async def verify_single_device(
    evaluator: Evaluator,
    parent_node,
    device: DeviceExtraction,
    idx1: int,
) -> None:
    """
    Build verification sub-tree and run checks for a single device.
    The structure strictly follows the rubric tree provided.
    """
    # Map index to canonical node names from rubric
    device_node_id = {1: "First_Streaming_Device", 2: "Second_Streaming_Device", 3: "Third_Streaming_Device"}[idx1]
    identity_node_id = f"Device_Identity_{idx1}"
    name_model_node_id = f"Device_Name_Model_{idx1}"
    mfg_node_id = f"Manufacturer_{idx1}"
    refurl_node_id = f"Reference_URL_{idx1}"

    video_node_id = f"Video_Specifications_{idx1}"
    k_node_id = f"4K_Support_{idx1}"
    hdr_node_id = f"HDR_Format_{idx1}"

    hw_node_id = f"Hardware_Performance_{idx1}"
    cpu_node_id = f"Processor_Spec_{idx1}"
    ram_node_id = f"RAM_Spec_{idx1}"
    storage_node_id = f"Storage_Spec_{idx1}"

    conn_node_id = f"Connectivity_{idx1}"
    wifi_node_id = f"WiFi_Standard_{idx1}"
    audio_node_id = f"Audio_Support_{idx1}"

    avail_node_id = f"Availability_Price_{idx1}"
    avail_leaf_id = f"Current_Availability_{idx1}"
    price_leaf_id = f"Price_Under_80_{idx1}"
    form_leaf_id = f"Form_Factor_{idx1}"

    urls = device.reference_urls or []
    label = device_label(device)

    # Device parent node (parallel, non-critical)
    device_node = evaluator.add_parallel(
        id=device_node_id,
        desc=f"{['First','Second','Third'][idx1-1]} streaming device meeting all technical specifications",
        parent=parent_node,
        critical=False,
    )

    # 1) Identity (critical group)
    identity_node = evaluator.add_parallel(
        id=identity_node_id,
        desc="Product identification including device name, model, manufacturer, and reference URL from manufacturer or major retailer",
        parent=device_node,
        critical=True,
    )

    # 1.a) Device name and model number must be provided (critical)
    has_name_and_model = has_text(device.device_name) and has_text(device.model_number)
    evaluator.add_custom_node(
        result=has_name_and_model,
        id=name_model_node_id,
        desc="Provide the complete device name and model number",
        parent=identity_node,
        critical=True,
    )

    # 1.b) Manufacturer must be provided (critical)
    evaluator.add_custom_node(
        result=has_text(device.manufacturer),
        id=mfg_node_id,
        desc="Provide the manufacturer name",
        parent=identity_node,
        critical=True,
    )

    # 1.c) Reference URL presence and validity (critical)
    # If no URLs, directly fail the node; else verify that at least one URL is from manufacturer or major retailer and is a product/spec page
    if not urls:
        evaluator.add_custom_node(
            result=False,
            id=refurl_node_id,
            desc="Provide a reference URL from the manufacturer's website or a major retailer (Amazon, Best Buy, Walmart, etc.) that confirms the device's existence and specifications",
            parent=identity_node,
            critical=True,
        )
    else:
        ref_leaf = evaluator.add_leaf(
            id=refurl_node_id,
            desc="Provide a reference URL from the manufacturer's website or a major retailer (Amazon, Best Buy, Walmart, etc.) that confirms the device's existence and specifications",
            parent=identity_node,
            critical=True,
        )
        ref_claim = (
            f"At least one of the provided pages is a legitimate product page for {label}, "
            f"hosted by either the device manufacturer or one of these major U.S. retailers: Amazon, Best Buy, or Walmart. "
            f"The page should present official product information or technical specifications."
        )
        await evaluator.verify(
            claim=ref_claim,
            node=ref_leaf,
            sources=urls,
            additional_instruction=(
                "Accept manufacturer domains (official brand sites) or the retailer domains: amazon.com, bestbuy.com, walmart.com. "
                "The page should clearly be a product page (not a forum/blog) and present specs or detailed product information."
            ),
        )

    # 2) Video Specifications (critical group)
    video_node = evaluator.add_parallel(
        id=video_node_id,
        desc="Video output capabilities including 4K resolution and HDR format support",
        parent=device_node,
        critical=True,
    )

    # 2.a) 4K support (critical)
    n4k = evaluator.add_leaf(
        id=k_node_id,
        desc="Device supports 4K Ultra HD video output (3840 x 2160 resolution)",
        parent=video_node,
        critical=True,
    )
    claim_4k = (
        f"The product page for {label} indicates support for 4K Ultra HD output, equivalent to 3840 × 2160 resolution "
        f"(may be phrased as 4K, Ultra HD, UHD, or 2160p; 'up to 4K' also qualifies)."
    )
    await evaluator.verify(
        claim=claim_4k,
        node=n4k,
        sources=urls,
        additional_instruction="Look for explicit mentions of 4K, Ultra HD, UHD, 2160p, or 3840×2160 on the product page.",
    )

    # 2.b) HDR support (critical) - at least one named format
    nhdr = evaluator.add_leaf(
        id=hdr_node_id,
        desc="Device supports at least one HDR format such as Dolby Vision, HDR10+, HDR10, or HLG",
        parent=video_node,
        critical=True,
    )
    claim_hdr = (
        f"The product page for {label} shows support for at least one HDR format among: Dolby Vision, HDR10+, HDR10, or HLG."
    )
    await evaluator.verify(
        claim=claim_hdr,
        node=nhdr,
        sources=urls,
        additional_instruction=(
            "Accept any of the following explicit strings (case-insensitive): 'Dolby Vision', 'HDR10+', 'HDR10', 'HLG', "
            "'Hybrid Log-Gamma'. Generic 'HDR' alone is insufficient without a named format."
        ),
    )

    # 3) Hardware Performance (critical group)
    hw_node = evaluator.add_parallel(
        id=hw_node_id,
        desc="Hardware specifications including processor, RAM, and storage",
        parent=device_node,
        critical=True,
    )

    # 3.a) Processor: at least quad-core and >= 1.5 GHz (critical)
    cpu_leaf = evaluator.add_leaf(
        id=cpu_node_id,
        desc="Device has at least a quad-core processor with minimum 1.5 GHz clock speed",
        parent=hw_node,
        critical=True,
    )
    claim_cpu = (
        f"The product page for {label} specifies a processor that is at least quad-core and clocked at 1.5 GHz or higher."
    )
    await evaluator.verify(
        claim=claim_cpu,
        node=cpu_leaf,
        sources=urls,
        additional_instruction=(
            "Both conditions must be satisfied: (1) core count >= 4 (quad-core or above), and (2) clock speed >= 1.5 GHz. "
            "If either is missing or not specified, the claim is not supported."
        ),
    )

    # 3.b) RAM >= 1 GB (critical)
    ram_leaf = evaluator.add_leaf(
        id=ram_node_id,
        desc="Device has at least 1GB of RAM",
        parent=hw_node,
        critical=True,
    )
    claim_ram = f"The product page for {label} indicates the device has at least 1 GB of RAM."
    await evaluator.verify(
        claim=claim_ram,
        node=ram_leaf,
        sources=urls,
        additional_instruction="Accept 1 GB, 2 GB, 3 GB, etc. If RAM capacity is not stated, the claim is not supported.",
    )

    # 3.c) Storage >= 8 GB (critical)
    storage_leaf = evaluator.add_leaf(
        id=storage_node_id,
        desc="Device provides at least 8GB of internal storage",
        parent=hw_node,
        critical=True,
    )
    claim_storage = f"The product page for {label} indicates at least 8 GB of internal storage."
    await evaluator.verify(
        claim=claim_storage,
        node=storage_leaf,
        sources=urls,
        additional_instruction="Accept 8 GB or higher. If internal storage is not stated, the claim is not supported.",
    )

    # 4) Connectivity (critical group)
    conn_node = evaluator.add_parallel(
        id=conn_node_id,
        desc="Wireless and audio connectivity capabilities",
        parent=device_node,
        critical=True,
    )

    # 4.a) Wi‑Fi 5 (802.11ac) or newer (critical)
    wifi_leaf = evaluator.add_leaf(
        id=wifi_node_id,
        desc="Device supports Wi-Fi 5 (802.11ac) or newer standard such as Wi-Fi 6 (802.11ax) or Wi-Fi 6E",
        parent=conn_node,
        critical=True,
    )
    claim_wifi = (
        f"The product page for {label} shows support for Wi‑Fi 5 (802.11ac) or a newer standard such as Wi‑Fi 6/6E (802.11ax)."
    )
    await evaluator.verify(
        claim=claim_wifi,
        node=wifi_leaf,
        sources=urls,
        additional_instruction="Look for '802.11ac' (Wi‑Fi 5) or '802.11ax' (Wi‑Fi 6 / 6E). Dual-band alone is insufficient.",
    )

    # 4.b) Audio: Dolby Atmos or DTS (critical)
    audio_leaf = evaluator.add_leaf(
        id=audio_node_id,
        desc="Device supports Dolby Atmos or DTS audio technology",
        parent=conn_node,
        critical=True,
    )
    claim_audio = (
        f"The product page for {label} shows support for at least one of these audio technologies: Dolby Atmos or DTS."
    )
    await evaluator.verify(
        claim=claim_audio,
        node=audio_leaf,
        sources=urls,
        additional_instruction="Accept explicit mentions of 'Dolby Atmos' or 'DTS' (including variants such as DTS:X).",
    )

    # 5) Availability & Price (critical group)
    avail_node = evaluator.add_parallel(
        id=avail_node_id,
        desc="Current market availability and pricing",
        parent=device_node,
        critical=True,
    )

    # 5.a) Currently available for purchase as of March 2026 (critical)
    avail_leaf = evaluator.add_leaf(
        id=avail_leaf_id,
        desc=f"Device is currently available for purchase as of {CURRENT_MONTH_YEAR} from major retailers or the manufacturer",
        parent=avail_node,
        critical=True,
    )
    claim_avail = (
        f"As of {CURRENT_MONTH_YEAR}, the product page for {label} indicates the device is available to purchase "
        f"(e.g., 'In stock', 'Add to cart', 'Buy now', or otherwise clearly for sale)."
    )
    await evaluator.verify(
        claim=claim_avail,
        node=avail_leaf,
        sources=urls,
        additional_instruction=(
            "If the page shows it is discontinued, out of stock with no purchase option, or otherwise not for sale, the claim is not supported."
        ),
    )

    # 5.b) Standard retail price under $80 (critical)
    price_leaf = evaluator.add_leaf(
        id=price_leaf_id,
        desc="Device's standard retail price is under $80 USD (excluding temporary sales or promotions)",
        parent=avail_node,
        critical=True,
    )
    claim_price = (
        f"The standard, regular (non-promotional) price or MSRP for {label} is under $80 USD."
    )
    await evaluator.verify(
        claim=claim_price,
        node=price_leaf,
        sources=urls,
        additional_instruction=(
            "Use MSRP, 'List Price', or the regular price. Do not rely solely on limited-time discounts or coupons. "
            "If only a sale price appears without a clear regular price, and no evidence the standard price is < $80, the claim is not supported."
        ),
    )

    # 5.c) Form factor is stick or compact device (critical)
    form_leaf = evaluator.add_leaf(
        id=form_leaf_id,
        desc="Device is a streaming stick or compact streaming device, not a full-size set-top box",
        parent=avail_node,
        critical=True,
    )
    claim_form = (
        f"The product is a streaming stick or compact streaming device (e.g., dongle, small puck). "
        f"It is not a full-size set-top box."
    )
    await evaluator.verify(
        claim=claim_form,
        node=form_leaf,
        sources=urls,
        additional_instruction=(
            "Accept descriptors like 'stick', 'dongle', 'compact', 'mini', 'puck'. "
            "If the device is a large set-top box or clearly not compact, the claim is not supported."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the streaming devices task against the rubric.
    """
    # Initialize evaluator (root: parallel as per rubric)
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
        default_model=model,
    )

    # Extract up to 3 devices from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_devices(),
        template_class=DevicesExtraction,
        extraction_name="devices_extraction",
    )

    devices = first_n_devices(extraction, 3)

    # Build and verify each device subtree
    for idx, device in enumerate(devices, start=1):
        await verify_single_device(evaluator, root, device, idx)

    # Optional: record basic stats
    evaluator.add_custom_info(
        info={"extracted_devices_count": len(extraction.devices) if extraction and extraction.devices else 0},
        info_type="extraction_stats",
        info_name="extraction_stats",
    )

    # Return structured evaluation summary
    return evaluator.get_summary()