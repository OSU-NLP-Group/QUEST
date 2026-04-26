import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "complete_gaming_workstation_setup"
TASK_DESCRIPTION = (
    "Provide four items (gaming laptop, external monitor, mechanical keyboard, wireless gaming mouse) "
    "meeting all stated technical and purchase requirements."
)

ALLOWED_RETAILERS = ["Best Buy", "Amazon", "Newegg", "Micro Center"]
ALLOWED_DOMAINS = {
    "bestbuy.com": "Best Buy",
    "amazon.com": "Amazon",
    "newegg.com": "Newegg",
    "microcenter.com": "Micro Center",
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ItemBase(BaseModel):
    product_name: Optional[str] = None
    model: Optional[str] = None
    retailer: Optional[str] = None
    product_url: Optional[str] = None
    price: Optional[str] = None
    stock_status: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


class LaptopSpecs(BaseModel):
    gpu: Optional[str] = None
    ram: Optional[str] = None
    storage: Optional[str] = None
    display_refresh_rate: Optional[str] = None
    warranty: Optional[str] = None


class MonitorSpecs(BaseModel):
    size_inches: Optional[str] = None
    refresh_rate: Optional[str] = None
    response_time_ms: Optional[str] = None
    color_gamut: Optional[str] = None  # e.g., "99% sRGB", "100% sRGB"


class KeyboardSpecs(BaseModel):
    keyboard_type: Optional[str] = None  # e.g., "mechanical"
    switch_type: Optional[str] = None  # e.g., "Linear", "Tactile", "Clicky", "Cherry MX Red"
    layout: Optional[str] = None  # e.g., "Full-size", "104-key", "100%", "TKL", "Tenkeyless", "80%"


class MouseSpecs(BaseModel):
    connectivity: Optional[str] = None  # e.g., "Wireless 2.4GHz", "Bluetooth"
    max_dpi: Optional[str] = None
    programmable_buttons: Optional[str] = None


class LaptopItem(ItemBase):
    specs: LaptopSpecs = Field(default_factory=LaptopSpecs)


class MonitorItem(ItemBase):
    specs: MonitorSpecs = Field(default_factory=MonitorSpecs)


class KeyboardItem(ItemBase):
    specs: KeyboardSpecs = Field(default_factory=KeyboardSpecs)


class MouseItem(ItemBase):
    specs: MouseSpecs = Field(default_factory=MouseSpecs)


class WorkstationExtraction(BaseModel):
    laptop: Optional[LaptopItem] = None
    monitor: Optional[MonitorItem] = None
    keyboard: Optional[KeyboardItem] = None
    mouse: Optional[MouseItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_workstation_items() -> str:
    return """
    Extract the four required items from the answer: gaming laptop, external monitor, mechanical keyboard, and wireless gaming mouse.
    For each item, extract the following fields exactly as presented in the answer text. Do not invent any values.

    Common fields for all items:
    - product_name: The product name and model as written by the answer
    - model: If a separate model identifier is provided; otherwise null
    - retailer: The retailer name (e.g., Best Buy, Amazon, Newegg, Micro Center)
    - product_url: A direct product page URL where the item can be purchased
    - price: The current price as stated (include currency symbol if present)
    - stock_status: The availability status stated (e.g., "In stock", "Available", "Out of stock")
    - additional_urls: Any other URLs the answer cites for this item (besides the direct product page), if any

    Laptop-specific fields (specs):
    - gpu: The GPU name (e.g., "NVIDIA GeForce RTX 5070")
    - ram: The RAM specification (e.g., "16GB")
    - storage: The storage type/specification (e.g., "1TB NVMe SSD")
    - display_refresh_rate: The display refresh rate (e.g., "144Hz", "165Hz")
    - warranty: The warranty information stated for the laptop (e.g., "1-year manufacturer warranty")

    Monitor-specific fields (specs):
    - size_inches: The screen size (e.g., "24-inch", "27\"")
    - refresh_rate: The refresh rate (e.g., "144Hz", "240Hz")
    - response_time_ms: The response time (e.g., "1ms", "5ms")
    - color_gamut: The color space coverage (e.g., "99% sRGB", "100% sRGB")

    Keyboard-specific fields (specs):
    - keyboard_type: The type (e.g., "mechanical", "membrane")
    - switch_type: The switch type or family (e.g., "Linear", "Tactile", "Clicky", "Cherry MX Red", "Gateron Brown")
    - layout: The keyboard layout (e.g., "Full-size", "104-key", "100%", "TKL", "Tenkeyless", "80%")

    Mouse-specific fields (specs):
    - connectivity: The connectivity (e.g., "Wireless 2.4GHz", "Bluetooth"; not just "wired-only")
    - max_dpi: The maximum DPI (e.g., "800", "16000")
    - programmable_buttons: The number of programmable buttons (e.g., "5", "6", "8")

    Return a JSON object with this structure:
    {
      "laptop": { ... },
      "monitor": { ... },
      "keyboard": { ... },
      "mouse": { ... }
    }

    If any item is missing in the answer, set that item to null.
    If a field is not mentioned, set it to null. For lists, return empty list when none.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def get_sources_list(item: ItemBase) -> List[str]:
    urls = []
    if item.product_url and item.product_url.strip():
        urls.append(item.product_url.strip())
    for u in item.additional_urls:
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())
    return urls


def url_domain(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    m = re.search(r"https?://([^/]+)", url)
    return m.group(1).lower() if m else None


def is_allowed_retailer(url: Optional[str], retailer_name: Optional[str]) -> bool:
    dom = url_domain(url) or ""
    allowed_by_domain = any(d in dom for d in ALLOWED_DOMAINS.keys())
    allowed_by_name = (retailer_name or "").strip() in ALLOWED_RETAILERS
    return allowed_by_domain or allowed_by_name


def additional_instruction_specs() -> str:
    return (
        "Use the product page(s) to verify the specific technical specification exactly. "
        "Allow reasonable synonyms (e.g., 'GeForce RTX' vs 'NVIDIA RTX'). "
        "For threshold checks (>= or <=), accept equivalent or higher/lower values. "
        "If multiple variants are shown, consider the primary configuration on the page."
    )


def additional_instruction_purchase() -> str:
    return (
        "Verify the purchase information from the product page: current price and stock availability. "
        "Price may include currency symbol and may be presented as a sale price. "
        "Stock status should indicate the item is available to purchase (e.g., 'In stock', 'Available for pickup')."
    )


def additional_instruction_keyboard_layout() -> str:
    return (
        "Verify the keyboard layout from the product page. Full-size can be described as '100%' or '104-key'. "
        "TKL can be described as 'tenkeyless' or '80%'. Consider reasonable synonyms."
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_laptop(evaluator: Evaluator, parent) -> None:
    extracted = evaluator._extraction_results[-1]["result"]  # Latest extraction result
    data = WorkstationExtraction(**extracted)
    laptop: Optional[LaptopItem] = data.laptop

    node_item = evaluator.add_sequential(
        id="Item_1_Gaming_Laptop",
        desc="Gaming laptop meeting all laptop technical and purchase requirements.",
        parent=parent,
        critical=False
    )

    # Identification: product name/model presence
    identification_ok = bool(laptop and laptop.product_name and laptop.product_name.strip())
    evaluator.add_custom_node(
        result=identification_ok,
        id="Laptop_Product_Identification",
        desc="Provide product name and model for the laptop.",
        parent=node_item,
        critical=True
    )

    # Technical specifications (critical group)
    node_specs = evaluator.add_parallel(
        id="Laptop_Technical_Specifications",
        desc="Laptop satisfies all required technical specifications.",
        parent=node_item,
        critical=True
    )

    sources = get_sources_list(laptop or ItemBase())

    # GPU >= RTX 5060
    gpu_claim = (
        f"The laptop's GPU is {laptop.specs.gpu} and it is NVIDIA GeForce RTX 5060 or better "
        f"(i.e., 5070/5080/5090)."
        if laptop and laptop.specs and laptop.specs.gpu else
        "The laptop's GPU is NVIDIA GeForce RTX 5060 or better (i.e., 5070/5080/5090)."
    )
    leaf_gpu = evaluator.add_leaf(
        id="GPU_Requirement",
        desc="Laptop GPU is NVIDIA RTX 5060 or better (e.g., RTX 5070/5080/5090).",
        parent=node_specs,
        critical=True
    )
    await evaluator.verify(
        claim=gpu_claim,
        node=leaf_gpu,
        sources=sources,
        additional_instruction=additional_instruction_specs()
    )

    # RAM >= 16GB
    ram_claim = "The laptop has at least 16GB of RAM."
    leaf_ram = evaluator.add_leaf(
        id="RAM_Requirement",
        desc="Laptop has at least 16GB RAM.",
        parent=node_specs,
        critical=True
    )
    await evaluator.verify(
        claim=ram_claim,
        node=leaf_ram,
        sources=sources,
        additional_instruction=additional_instruction_specs()
    )

    # NVMe SSD storage
    storage_claim = "The laptop storage uses NVMe SSD (not SATA SSD or HDD)."
    leaf_storage = evaluator.add_leaf(
        id="Storage_Requirement",
        desc="Laptop storage is NVMe SSD (not SATA SSD or HDD).",
        parent=node_specs,
        critical=True
    )
    await evaluator.verify(
        claim=storage_claim,
        node=leaf_storage,
        sources=sources,
        additional_instruction=additional_instruction_specs()
    )

    # Display refresh rate >= 144Hz
    display_claim = "The laptop display refresh rate is at least 144Hz."
    leaf_display = evaluator.add_leaf(
        id="Display_Refresh_Rate_Requirement",
        desc="Laptop display refresh rate is at least 144Hz.",
        parent=node_specs,
        critical=True
    )
    await evaluator.verify(
        claim=display_claim,
        node=leaf_display,
        sources=sources,
        additional_instruction=additional_instruction_specs()
    )

    # Purchase information (critical group)
    node_purchase = evaluator.add_parallel(
        id="Laptop_Purchase_Information",
        desc="Laptop purchase requirements are satisfied.",
        parent=node_item,
        critical=True
    )

    # Retailer requirement – custom check on domain/name
    evaluator.add_custom_node(
        result=is_allowed_retailer(laptop.product_url if laptop else None, laptop.retailer if laptop else None),
        id="Laptop_Retailer_Requirement",
        desc="Laptop is available from at least one of: Best Buy, Amazon, Newegg, Micro Center.",
        parent=node_purchase,
        critical=True
    )

    # Product page URL provided
    evaluator.add_custom_node(
        result=bool(laptop and laptop.product_url and laptop.product_url.strip()),
        id="Laptop_Product_Page_URL",
        desc="Provide a direct product page URL for purchase.",
        parent=node_purchase,
        critical=True
    )

    # Current price verification
    leaf_price = evaluator.add_leaf(
        id="Laptop_Current_Price",
        desc="Provide the current price.",
        parent=node_purchase,
        critical=True
    )
    price_claim = f"The current price shown on the product page is {laptop.price}." if laptop and laptop.price else \
        "The product page shows the current price."
    await evaluator.verify(
        claim=price_claim,
        node=leaf_price,
        sources=sources,
        additional_instruction=additional_instruction_purchase()
    )

    # Stock availability verification
    leaf_stock = evaluator.add_leaf(
        id="Laptop_Stock_Availability",
        desc="Confirm the item is currently in stock and available for purchase.",
        parent=node_purchase,
        critical=True
    )
    stock_claim = "The product page shows the laptop is in stock and available for purchase."
    await evaluator.verify(
        claim=stock_claim,
        node=leaf_stock,
        sources=sources,
        additional_instruction=additional_instruction_purchase()
    )

    # Warranty at least 1 year
    leaf_warranty = evaluator.add_leaf(
        id="Laptop_Warranty_Information",
        desc="Provide/confirm at least a 1-year manufacturer warranty for the laptop.",
        parent=node_purchase,
        critical=True
    )
    warranty_claim = "The laptop includes at least a 1-year manufacturer warranty."
    await evaluator.verify(
        claim=warranty_claim,
        node=leaf_warranty,
        sources=sources,
        additional_instruction="Verify the warranty info from the product page or manufacturer's listing. "
                               "Accept '1 year', '12 months', or longer durations."
    )


async def verify_monitor(evaluator: Evaluator, parent) -> None:
    extracted = evaluator._extraction_results[-1]["result"]
    data = WorkstationExtraction(**extracted)
    monitor: Optional[MonitorItem] = data.monitor

    node_item = evaluator.add_sequential(
        id="Item_2_External_Monitor",
        desc="External monitor meeting all monitor technical and purchase requirements.",
        parent=parent,
        critical=False
    )

    identification_ok = bool(monitor and monitor.product_name and monitor.product_name.strip())
    evaluator.add_custom_node(
        result=identification_ok,
        id="Monitor_Product_Identification",
        desc="Provide product name and model for the monitor.",
        parent=node_item,
        critical=True
    )

    node_specs = evaluator.add_parallel(
        id="Monitor_Technical_Specifications",
        desc="Monitor satisfies all required technical specifications.",
        parent=node_item,
        critical=True
    )

    sources = get_sources_list(monitor or ItemBase())

    # Size >= 24 inches
    leaf_size = evaluator.add_leaf(
        id="Screen_Size_Requirement",
        desc="Monitor screen size is at least 24 inches diagonal.",
        parent=node_specs,
        critical=True
    )
    size_claim = "The monitor has a screen size of at least 24 inches."
    await evaluator.verify(
        claim=size_claim,
        node=leaf_size,
        sources=sources,
        additional_instruction=additional_instruction_specs()
    )

    # Refresh rate >= 144Hz
    leaf_rr = evaluator.add_leaf(
        id="Refresh_Rate_Requirement",
        desc="Monitor refresh rate is at least 144Hz.",
        parent=node_specs,
        critical=True
    )
    rr_claim = "The monitor refresh rate is at least 144Hz."
    await evaluator.verify(
        claim=rr_claim,
        node=leaf_rr,
        sources=sources,
        additional_instruction=additional_instruction_specs()
    )

    # Response time <= 5ms
    leaf_rt = evaluator.add_leaf(
        id="Response_Time_Requirement",
        desc="Monitor response time is 5ms or less.",
        parent=node_specs,
        critical=True
    )
    rt_claim = "The monitor response time is 5ms or less."
    await evaluator.verify(
        claim=rt_claim,
        node=leaf_rt,
        sources=sources,
        additional_instruction=additional_instruction_specs()
    )

    # Color accuracy >= 99% sRGB
    leaf_color = evaluator.add_leaf(
        id="Color_Accuracy_Requirement",
        desc="Monitor supports at least 99% sRGB coverage.",
        parent=node_specs,
        critical=True
    )
    color_claim = "The monitor supports at least 99% sRGB color gamut coverage."
    await evaluator.verify(
        claim=color_claim,
        node=leaf_color,
        sources=sources,
        additional_instruction=additional_instruction_specs()
    )

    node_purchase = evaluator.add_parallel(
        id="Monitor_Purchase_Information",
        desc="Monitor purchase requirements are satisfied.",
        parent=node_item,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_allowed_retailer(monitor.product_url if monitor else None, monitor.retailer if monitor else None),
        id="Monitor_Retailer_Requirement",
        desc="Monitor is available from at least one of: Best Buy, Amazon, Newegg, Micro Center.",
        parent=node_purchase,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(monitor and monitor.product_url and monitor.product_url.strip()),
        id="Monitor_Product_Page_URL",
        desc="Provide a direct product page URL for purchase.",
        parent=node_purchase,
        critical=True
    )

    leaf_price = evaluator.add_leaf(
        id="Monitor_Current_Price",
        desc="Provide the current price.",
        parent=node_purchase,
        critical=True
    )
    price_claim = f"The current price shown on the product page is {monitor.price}." if monitor and monitor.price else \
        "The product page shows the current price."
    await evaluator.verify(
        claim=price_claim,
        node=leaf_price,
        sources=sources,
        additional_instruction=additional_instruction_purchase()
    )

    leaf_stock = evaluator.add_leaf(
        id="Monitor_Stock_Availability",
        desc="Confirm the item is currently in stock and available for purchase.",
        parent=node_purchase,
        critical=True
    )
    stock_claim = "The product page shows the monitor is in stock and available for purchase."
    await evaluator.verify(
        claim=stock_claim,
        node=leaf_stock,
        sources=sources,
        additional_instruction=additional_instruction_purchase()
    )


async def verify_keyboard(evaluator: Evaluator, parent) -> None:
    extracted = evaluator._extraction_results[-1]["result"]
    data = WorkstationExtraction(**extracted)
    keyboard: Optional[KeyboardItem] = data.keyboard

    node_item = evaluator.add_sequential(
        id="Item_3_Mechanical_Keyboard",
        desc="Mechanical keyboard meeting all keyboard technical and purchase requirements.",
        parent=parent,
        critical=False
    )

    identification_ok = bool(keyboard and keyboard.product_name and keyboard.product_name.strip())
    evaluator.add_custom_node(
        result=identification_ok,
        id="Keyboard_Product_Identification",
        desc="Provide product name and model for the keyboard.",
        parent=node_item,
        critical=True
    )

    node_specs = evaluator.add_parallel(
        id="Keyboard_Technical_Specifications",
        desc="Keyboard satisfies all required technical specifications.",
        parent=node_item,
        critical=True
    )

    sources = get_sources_list(keyboard or ItemBase())

    # Mechanical type
    leaf_mech = evaluator.add_leaf(
        id="Mechanical_Keyboard_Requirement",
        desc="Keyboard is mechanical (not membrane).",
        parent=node_specs,
        critical=True
    )
    mech_claim = "The keyboard is mechanical (not membrane)."
    await evaluator.verify(
        claim=mech_claim,
        node=leaf_mech,
        sources=sources,
        additional_instruction=additional_instruction_specs()
    )

    # Switch type specified – custom existence check
    switch_specified = bool(keyboard and keyboard.specs and keyboard.specs.switch_type and keyboard.specs.switch_type.strip())
    evaluator.add_custom_node(
        result=switch_specified,
        id="Switch_Type_Specified_Requirement",
        desc="Switch type is specified (Linear, Tactile, or Clicky).",
        parent=node_specs,
        critical=True
    )

    # Layout full-size or TKL – verify on page
    leaf_layout = evaluator.add_leaf(
        id="Layout_Requirement",
        desc="Keyboard layout is full-size (100%/104-key) or TKL (80%).",
        parent=node_specs,
        critical=True
    )
    layout_claim = "The keyboard layout is full-size (100%/104-key) or TKL (tenkeyless/80%)."
    await evaluator.verify(
        claim=layout_claim,
        node=leaf_layout,
        sources=sources,
        additional_instruction=additional_instruction_keyboard_layout()
    )

    node_purchase = evaluator.add_parallel(
        id="Keyboard_Purchase_Information",
        desc="Keyboard purchase requirements are satisfied.",
        parent=node_item,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_allowed_retailer(keyboard.product_url if keyboard else None, keyboard.retailer if keyboard else None),
        id="Keyboard_Retailer_Requirement",
        desc="Keyboard is available from at least one of: Best Buy, Amazon, Newegg, Micro Center.",
        parent=node_purchase,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(keyboard and keyboard.product_url and keyboard.product_url.strip()),
        id="Keyboard_Product_Page_URL",
        desc="Provide a direct product page URL for purchase.",
        parent=node_purchase,
        critical=True
    )

    leaf_price = evaluator.add_leaf(
        id="Keyboard_Current_Price",
        desc="Provide the current price.",
        parent=node_purchase,
        critical=True
    )
    price_claim = f"The current price shown on the product page is {keyboard.price}." if keyboard and keyboard.price else \
        "The product page shows the current price."
    await evaluator.verify(
        claim=price_claim,
        node=leaf_price,
        sources=sources,
        additional_instruction=additional_instruction_purchase()
    )

    leaf_stock = evaluator.add_leaf(
        id="Keyboard_Stock_Availability",
        desc="Confirm the item is currently in stock and available for purchase.",
        parent=node_purchase,
        critical=True
    )
    stock_claim = "The product page shows the keyboard is in stock and available for purchase."
    await evaluator.verify(
        claim=stock_claim,
        node=leaf_stock,
        sources=sources,
        additional_instruction=additional_instruction_purchase()
    )


async def verify_mouse(evaluator: Evaluator, parent) -> None:
    extracted = evaluator._extraction_results[-1]["result"]
    data = WorkstationExtraction(**extracted)
    mouse: Optional[MouseItem] = data.mouse

    node_item = evaluator.add_sequential(
        id="Item_4_Wireless_Gaming_Mouse",
        desc="Wireless gaming mouse meeting all mouse technical and purchase requirements.",
        parent=parent,
        critical=False
    )

    identification_ok = bool(mouse and mouse.product_name and mouse.product_name.strip())
    evaluator.add_custom_node(
        result=identification_ok,
        id="Mouse_Product_Identification",
        desc="Provide product name and model for the mouse.",
        parent=node_item,
        critical=True
    )

    node_specs = evaluator.add_parallel(
        id="Mouse_Technical_Specifications",
        desc="Mouse satisfies all required technical specifications.",
        parent=node_item,
        critical=True
    )

    sources = get_sources_list(mouse or ItemBase())

    # Wireless connectivity
    leaf_wireless = evaluator.add_leaf(
        id="Wireless_Connectivity_Requirement",
        desc="Mouse is wireless (2.4GHz wireless or Bluetooth; not wired-only).",
        parent=node_specs,
        critical=True
    )
    wireless_claim = "The mouse is wireless using 2.4GHz and/or Bluetooth (not wired-only)."
    await evaluator.verify(
        claim=wireless_claim,
        node=leaf_wireless,
        sources=sources,
        additional_instruction=additional_instruction_specs()
    )

    # Max DPI >= 800
    leaf_dpi = evaluator.add_leaf(
        id="DPI_Requirement",
        desc="Mouse has adjustable DPI with maximum DPI of at least 800.",
        parent=node_specs,
        critical=True
    )
    dpi_claim = "The mouse has adjustable DPI with a maximum DPI of at least 800."
    await evaluator.verify(
        claim=dpi_claim,
        node=leaf_dpi,
        sources=sources,
        additional_instruction=additional_instruction_specs()
    )

    # Programmable buttons >= 5
    leaf_buttons = evaluator.add_leaf(
        id="Programmable_Buttons_Requirement",
        desc="Mouse has at least 5 programmable buttons.",
        parent=node_specs,
        critical=True
    )
    buttons_claim = "The mouse has at least 5 programmable buttons."
    await evaluator.verify(
        claim=buttons_claim,
        node=leaf_buttons,
        sources=sources,
        additional_instruction=additional_instruction_specs()
    )

    node_purchase = evaluator.add_parallel(
        id="Mouse_Purchase_Information",
        desc="Mouse purchase requirements are satisfied.",
        parent=node_item,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_allowed_retailer(mouse.product_url if mouse else None, mouse.retailer if mouse else None),
        id="Mouse_Retailer_Requirement",
        desc="Mouse is available from at least one of: Best Buy, Amazon, Newegg, Micro Center.",
        parent=node_purchase,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(mouse and mouse.product_url and mouse.product_url.strip()),
        id="Mouse_Product_Page_URL",
        desc="Provide a direct product page URL for purchase.",
        parent=node_purchase,
        critical=True
    )

    leaf_price = evaluator.add_leaf(
        id="Mouse_Current_Price",
        desc="Provide the current price.",
        parent=node_purchase,
        critical=True
    )
    price_claim = f"The current price shown on the product page is {mouse.price}." if mouse and mouse.price else \
        "The product page shows the current price."
    await evaluator.verify(
        claim=price_claim,
        node=leaf_price,
        sources=sources,
        additional_instruction=additional_instruction_purchase()
    )

    leaf_stock = evaluator.add_leaf(
        id="Mouse_Stock_Availability",
        desc="Confirm the item is currently in stock and available for purchase.",
        parent=node_purchase,
        critical=True
    )
    stock_claim = "The product page shows the mouse is in stock and available for purchase."
    await evaluator.verify(
        claim=stock_claim,
        node=leaf_stock,
        sources=sources,
        additional_instruction=additional_instruction_purchase()
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Complete Gaming Workstation Setup task.
    """
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

    # Extract all items from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_workstation_items(),
        template_class=WorkstationExtraction,
        extraction_name="workstation_items"
    )

    # Build verification tree under root
    # Add a top-level non-critical aggregator to allow partial credit across items
    setup_node = evaluator.add_parallel(
        id="Complete_Gaming_Workstation_Setup",
        desc=TASK_DESCRIPTION,
        parent=root,
        critical=False
    )

    # Verify each item sub-tree
    await verify_laptop(evaluator, setup_node)
    await verify_monitor(evaluator, setup_node)
    await verify_keyboard(evaluator, setup_node)
    await verify_mouse(evaluator, setup_node)

    # Add custom info for allowed retailers/domains
    evaluator.add_custom_info(
        info={"allowed_retailers": ALLOWED_RETAILERS, "allowed_domains": list(ALLOWED_DOMAINS.keys())},
        info_type="constraints",
        info_name="purchase_constraints"
    )

    return evaluator.get_summary()