import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gaming_laptops_compare"
TASK_DESCRIPTION = """
I am looking to purchase a gaming laptop for moderate to high-performance gaming and need to compare options. Please identify four different gaming laptops currently available for purchase that meet ALL of the following requirements:

Display Requirements:
- Screen size must be at least 15.6 inches
- Resolution must be at least Full HD (1920×1080)
- Refresh rate must be at least 120Hz

Performance Requirements:
- Processor must be Intel Core i7 (13th Generation or newer) OR AMD Ryzen 7 (7000 series or newer)
- Graphics card must be NVIDIA GeForce RTX 4060 or better
- RAM must be at least 16GB
- Storage must be at least 512GB SSD

Connectivity Requirements:
- Must have at least one USB-C port with USB 3.2 or higher standard (supporting 10Gbps or faster data transfer)

Portability and Price Requirements:
- Weight must be 2.5kg (5.5 lbs) or less
- Price must be between $1,200 and $2,000 USD

For each laptop, provide:
1. The manufacturer name and full model name
2. All key specifications (display size/resolution/refresh rate, processor model, GPU model, RAM amount, storage capacity, USB-C specification, weight)
3. Battery capacity in Wh OR estimated battery life in hours
4. Current price
5. A direct URL to a retailer (such as Amazon, Best Buy, Newegg, or manufacturer's store) where the laptop is currently in stock and available for purchase
6. A direct URL to the official manufacturer's product page showing the complete specifications
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LaptopItem(BaseModel):
    """Structured fields for one laptop as provided by the agent answer."""
    manufacturer: Optional[str] = None
    model_name: Optional[str] = None

    display_size: Optional[str] = None          # e.g., "15.6-inch", "16”"
    resolution: Optional[str] = None            # e.g., "1920x1080", "QHD 2560x1440"
    refresh_rate: Optional[str] = None          # e.g., "144Hz", "120 Hz"

    processor_model: Optional[str] = None       # e.g., "Intel Core i7-13700H", "AMD Ryzen 7 7840HS"
    gpu_model: Optional[str] = None             # e.g., "NVIDIA GeForce RTX 4060 Laptop GPU"
    ram: Optional[str] = None                   # e.g., "16GB", "32 GB DDR5"
    storage: Optional[str] = None               # e.g., "1TB SSD", "512GB PCIe NVMe SSD"

    usb_c_spec: Optional[str] = None            # e.g., "USB-C 3.2 Gen 2 (10Gbps)", "Thunderbolt 4"
    weight: Optional[str] = None                # e.g., "2.3 kg", "5.1 lbs"

    battery_wh: Optional[str] = None            # e.g., "90Wh"
    battery_life_hours: Optional[str] = None    # e.g., "8 hours"

    price: Optional[str] = None                 # e.g., "$1,499", "USD 1,699"
    retailer_url: Optional[str] = None
    product_page_url: Optional[str] = None


class LaptopsExtraction(BaseModel):
    """Extraction of all laptops in the answer."""
    items: List[LaptopItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_laptops() -> str:
    return """
    Extract up to four gaming laptops from the answer. For each laptop, extract the following fields exactly as presented:

    - manufacturer: Manufacturer/brand name
    - model_name: Full model name/number
    - display_size: Screen size (with unit if provided)
    - resolution: Screen resolution string (e.g., "1920x1080", "QHD 2560x1440")
    - refresh_rate: Display refresh rate (e.g., "120Hz", "165 Hz")
    - processor_model: CPU model string (e.g., "Intel Core i7-13700H", "AMD Ryzen 7 7840HS")
    - gpu_model: GPU model string (e.g., "NVIDIA GeForce RTX 4060")
    - ram: RAM amount string (e.g., "16GB", "32 GB")
    - storage: Storage capacity/type string (e.g., "512GB SSD", "1TB NVMe SSD")
    - usb_c_spec: USB-C specification string relevant to data rate (e.g., "USB 3.2 Gen 2", "Thunderbolt 4", "USB4")
    - weight: Weight with unit (e.g., "2.4 kg", "5.2 lbs")
    - battery_wh: Battery capacity in Wh if provided; otherwise null
    - battery_life_hours: Estimated battery life in hours if provided; otherwise null
    - price: Current price string (with currency if provided)
    - retailer_url: Direct URL to a retailer product page that is currently selling this model
    - product_page_url: Direct URL to the official manufacturer product page showing specifications

    Rules:
    - Extract only what is explicitly in the answer; do not invent or infer.
    - If more than four laptops are provided, keep the first four in the order they appear.
    - If a field is missing for a laptop, return null for that field.
    - Ensure URLs are valid and complete; if a URL is missing protocol, prepend http://.
    - Do not include comparison commentary or extra fields.
    Return a JSON object with a single 'items' array containing up to four laptop objects with the fields listed above.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s) and bool(str(s).strip())

def _srcs(*urls: Optional[str]) -> List[str]:
    return [u for u in urls if _non_empty(u)]

# --------------------------------------------------------------------------- #
# Verification per laptop                                                     #
# --------------------------------------------------------------------------- #
async def verify_one_laptop(
    evaluator: Evaluator,
    parent_node,
    laptop: LaptopItem,
    index: int,
) -> None:
    """
    Build the verification subtree for a single laptop and perform checks.
    """
    lap_idx = index + 1
    laptop_node = evaluator.add_parallel(
        id=f"laptop_{lap_idx}",
        desc=f"Laptop #{lap_idx}: verification of requirements and sources",
        parent=parent_node,
        critical=False  # Allow partial credit per laptop
    )

    # Create a sequential pipeline under this laptop to gate later checks on early failures
    pipeline_node = evaluator.add_sequential(
        id=f"laptop_{lap_idx}_pipeline",
        desc=f"Laptop #{lap_idx}: pipeline (presence → identity match → constraints)",
        parent=laptop_node,
        critical=False
    )

    # 0) Presence checks (critical group) — if any fails, subsequent checks are skipped
    presence_node = evaluator.add_parallel(
        id=f"laptop_{lap_idx}_presence",
        desc=f"Laptop #{lap_idx}: required fields presence",
        parent=pipeline_node,
        critical=True  # The group itself is gating for the sequential pipeline
    )

    evaluator.add_custom_node(
        result=_non_empty(laptop.manufacturer) and _non_empty(laptop.model_name),
        id=f"laptop_{lap_idx}_identification_present",
        desc="Manufacturer and full model name are provided in the answer",
        parent=presence_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(laptop.retailer_url),
        id=f"laptop_{lap_idx}_retailer_url_present",
        desc="Retailer URL is provided",
        parent=presence_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(laptop.product_page_url),
        id=f"laptop_{lap_idx}_product_url_present",
        desc="Official manufacturer product page URL is provided",
        parent=presence_node,
        critical=True
    )

    # 1) Identification: confirm manufacturer and model match the product page
    id_match_node = evaluator.add_leaf(
        id=f"laptop_{lap_idx}_identification",
        desc="Manufacturer and model match the official product page",
        parent=pipeline_node,
        critical=True
    )
    id_claim = (
        f"The official product page corresponds to the laptop manufactured by '{laptop.manufacturer}' "
        f"with model '{laptop.model_name}'."
    )
    await evaluator.verify(
        claim=id_claim,
        node=id_match_node,
        sources=laptop.product_page_url,
        additional_instruction=(
            "Confirm the page is for the same product model. Allow minor formatting differences, suffixes like (2024/2025), "
            "regional variants, and case differences, as long as it is clearly the same model from the same manufacturer."
        )
    )

    # 2) Constraints group (parallel) — each sub-check critical; failures here will stop subsequent siblings in sequential pipeline
    constraints_node = evaluator.add_parallel(
        id=f"laptop_{lap_idx}_constraints",
        desc=f"Laptop #{lap_idx}: technical and commercial constraints",
        parent=pipeline_node,
        critical=False
    )

    # 2a) Display group
    display_node = evaluator.add_parallel(
        id=f"laptop_{lap_idx}_display",
        desc="Display must meet size, resolution, and refresh rate requirements",
        parent=constraints_node,
        critical=False
    )

    size_leaf = evaluator.add_leaf(
        id=f"laptop_{lap_idx}_display_size",
        desc="Display size is at least 15.6 inches",
        parent=display_node,
        critical=True
    )
    await evaluator.verify(
        claim="This laptop's display size is at least 15.6 inches.",
        node=size_leaf,
        sources=_srcs(laptop.product_page_url, laptop.retailer_url),
        additional_instruction="Use the specification section. If inches not shown, convert from cm (≥ 39.6 cm). Accept 16-inch, 17-inch, etc."
    )

    resolution_leaf = evaluator.add_leaf(
        id=f"laptop_{lap_idx}_display_resolution",
        desc="Resolution is at least Full HD (1920×1080)",
        parent=display_node,
        critical=True
    )
    await evaluator.verify(
        claim="This laptop's display resolution is at least 1920×1080 (FHD) or higher (e.g., 2560×1440, 3840×2160).",
        node=resolution_leaf,
        sources=_srcs(laptop.product_page_url, laptop.retailer_url),
        additional_instruction="Accept terms FHD/Full HD, QHD/WQHD, 2K, 4K/UHD if pixel dimensions imply ≥1920×1080."
    )

    refresh_leaf = evaluator.add_leaf(
        id=f"laptop_{lap_idx}_display_refresh",
        desc="Refresh rate is at least 120Hz",
        parent=display_node,
        critical=True
    )
    await evaluator.verify(
        claim="This laptop's display refresh rate is at least 120 Hz.",
        node=refresh_leaf,
        sources=_srcs(laptop.product_page_url, laptop.retailer_url),
        additional_instruction="Accept 120Hz, 144Hz, 165Hz, 240Hz, etc. Ignore variable refresh marketing unless numeric ≥120Hz is present."
    )

    # 2b) Performance group
    perf_node = evaluator.add_parallel(
        id=f"laptop_{lap_idx}_performance",
        desc="Performance requirements (CPU/GPU/RAM/Storage/USB-C)",
        parent=constraints_node,
        critical=False
    )

    cpu_leaf = evaluator.add_leaf(
        id=f"laptop_{lap_idx}_processor",
        desc="CPU is Intel Core i7 (13th gen or newer) OR AMD Ryzen 7 (7000 series or newer)",
        parent=perf_node,
        critical=True
    )
    await evaluator.verify(
        claim="The processor is Intel Core i7 (13th generation or newer) OR AMD Ryzen 7 (7000 series or newer).",
        node=cpu_leaf,
        sources=_srcs(laptop.product_page_url, laptop.retailer_url),
        additional_instruction=(
            "Examples that PASS: i7-13650HX, i7-13700H, i7-14700HX, Ryzen 7 7840HS, 7745HX, 8845HS. "
            "Examples that FAIL: i7-12700H, Ryzen 7 5800H. Verify exact CPU on the page."
        )
    )

    gpu_leaf = evaluator.add_leaf(
        id=f"laptop_{lap_idx}_gpu",
        desc="GPU is NVIDIA GeForce RTX 4060 or better",
        parent=perf_node,
        critical=True
    )
    await evaluator.verify(
        claim="The graphics card is NVIDIA GeForce RTX 4060 or better (e.g., RTX 4070/4080/4090).",
        node=gpu_leaf,
        sources=_srcs(laptop.product_page_url, laptop.retailer_url),
        additional_instruction="Laptop GPU variants are acceptable. If only RTX 4050 is shown, this must FAIL."
    )

    ram_leaf = evaluator.add_leaf(
        id=f"laptop_{lap_idx}_ram",
        desc="RAM is at least 16GB",
        parent=perf_node,
        critical=True
    )
    await evaluator.verify(
        claim="The RAM capacity is 16 GB or higher.",
        node=ram_leaf,
        sources=_srcs(laptop.product_page_url, laptop.retailer_url),
        additional_instruction="Accept 16GB, 32GB, 64GB. If 8GB appears anywhere as the capacity for the specified model, FAIL."
    )

    storage_leaf = evaluator.add_leaf(
        id=f"laptop_{lap_idx}_storage",
        desc="Storage is at least 512GB SSD",
        parent=perf_node,
        critical=True
    )
    await evaluator.verify(
        claim="The primary storage is SSD with capacity of at least 512 GB.",
        node=storage_leaf,
        sources=_srcs(laptop.product_page_url, laptop.retailer_url),
        additional_instruction="Accept 512GB SSD, 1TB SSD, NVMe SSD. HDD-only or 256GB SSD should FAIL."
    )

    usbc_leaf = evaluator.add_leaf(
        id=f"laptop_{lap_idx}_usbc",
        desc="Has at least one USB-C port with USB 3.2 (10Gbps) or higher",
        parent=perf_node,
        critical=True
    )
    await evaluator.verify(
        claim="There is at least one USB‑C port supporting USB 3.2 Gen 2 (10 Gbps) or higher (USB 3.2 Gen 2x2, USB4, Thunderbolt 3/4).",
        node=usbc_leaf,
        sources=_srcs(laptop.product_page_url, laptop.retailer_url),
        additional_instruction=(
            "PASS if the page mentions USB‑C Gen 2 (10Gbps), Gen 2x2 (20Gbps), USB4, Thunderbolt 3/4. "
            "FAIL if only USB 3.2 Gen 1 (5Gbps) is present with no higher-speed USB‑C port."
        )
    )

    # 2c) Battery (non-critical – only presence requirement)
    battery_leaf = evaluator.add_custom_node(
        result=_non_empty(laptop.battery_wh) or _non_empty(laptop.battery_life_hours),
        id=f"laptop_{lap_idx}_battery",
        desc="Battery capacity (Wh) or battery life (hours) is provided in the answer",
        parent=constraints_node,
        critical=False  # Non-critical criterion
    )

    # 2d) Weight
    weight_leaf = evaluator.add_leaf(
        id=f"laptop_{lap_idx}_weight",
        desc="Weight is 2.5 kg (5.5 lbs) or less",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim="The laptop's weight is 2.5 kg (5.5 lbs) or less.",
        node=weight_leaf,
        sources=_srcs(laptop.product_page_url, laptop.retailer_url),
        additional_instruction="If weight is given in lbs, convert: 5.5 lbs ≈ 2.5 kg. PASS if ≤ 2.5 kg or ≤ 5.5 lbs."
    )

    # 2e) Price (retailer)
    price_leaf = evaluator.add_leaf(
        id=f"laptop_{lap_idx}_price",
        desc="Current price on the retailer page is between $1,200 and $2,000 USD",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim="The current price on the retailer page is between $1,200 and $2,000 USD for the specified model.",
        node=price_leaf,
        sources=laptop.retailer_url,
        additional_instruction=(
            "Use the current price shown (sale price acceptable). Ignore shipping/tax. "
            "The currency must be USD. If the page shows multiple configurations, judge the price for the same configuration referenced."
        )
    )

    # 2f) Availability (retailer)
    availability_leaf = evaluator.add_leaf(
        id=f"laptop_{lap_idx}_availability",
        desc="Retailer page shows the laptop is currently in stock/available for purchase",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim="The retailer page indicates the laptop is currently available to purchase (e.g., In Stock, Add to Cart, Buy Now).",
        node=availability_leaf,
        sources=laptop.retailer_url,
        additional_instruction="Look for clear purchase affordances ('Add to Cart', 'Buy Now') or 'In Stock'. Preorder counts as available."
    )

    # 3) Reference (official manufacturer product page validity)
    reference_leaf = evaluator.add_leaf(
        id=f"laptop_{lap_idx}_reference",
        desc="The provided official URL is the manufacturer's product page with complete specifications",
        parent=pipeline_node,
        critical=True
    )
    await evaluator.verify(
        claim="This URL is the official manufacturer product page for the specified model and contains a specifications section.",
        node=reference_leaf,
        sources=laptop.product_page_url,
        additional_instruction=(
            "PASS if the domain belongs to the manufacturer and the page includes specification details. "
            "Support pages with full specs also PASS. Unauthorized third-party pages FAIL."
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
    Evaluate the agent's answer for the gaming laptops comparison task.
    """
    # Initialize evaluator; make root non-critical to allow partial scoring across items
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

    # Extract up to four laptops
    extracted = await evaluator.extract(
        prompt=prompt_extract_laptops(),
        template_class=LaptopsExtraction,
        extraction_name="laptops_extraction"
    )

    # Record ground-truth constraint overview (for transparency)
    evaluator.add_ground_truth({
        "required_count": 4,
        "constraints": {
            "display": {"size_min_inch": 15.6, "resolution_min": "1920x1080", "refresh_min_hz": 120},
            "cpu": "Intel Core i7 13th gen+ OR AMD Ryzen 7 7000+",
            "gpu": "NVIDIA GeForce RTX 4060 or better",
            "ram_min_gb": 16,
            "storage_min_ssd_gb": 512,
            "usb_c": "≥ USB 3.2 Gen 2 (10Gbps) or higher",
            "weight_max_kg": 2.5,
            "price_usd_range": [1200, 2000],
            "availability": "Retailer page shows in-stock/available",
            "reference": "Official manufacturer product page with specs"
        }
    })

    # Prepare exactly four laptop entries (truncate or pad)
    items: List[LaptopItem] = list(extracted.items[:4])
    while len(items) < 4:
        items.append(LaptopItem())

    # Build verification for each laptop
    for idx, laptop in enumerate(items):
        await verify_one_laptop(evaluator, root, laptop, idx)

    # Return evaluation summary
    return evaluator.get_summary()