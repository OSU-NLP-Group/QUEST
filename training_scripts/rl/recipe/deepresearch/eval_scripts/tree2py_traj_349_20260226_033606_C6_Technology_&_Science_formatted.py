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
TASK_ID = "matter_devices_4"
TASK_DESCRIPTION = """
Find four Matter-compatible smart home devices, each from a different device category (such as smart lighting, smart locks, sensors, smart plugs, switches, thermostats, or cameras). For each device, provide the following information:
(1) Device Category: Clearly identify which smart home device category the product belongs to.
(2) Matter Compatibility: Confirm that the device is officially Matter-compatible or Matter-certified, with a reference URL for verification.
(3) Ecosystem Compatibility: Identify at least one major smart home ecosystem (Apple Home, Google Home, or Amazon Alexa) that the device is compatible with, and provide a reference URL for verification.
(4) Technical Specifications: Include the device's connectivity protocol (Thread, Wi-Fi, Bluetooth, or Ethernet) and at least one additional technical specification (such as power requirements, dimensions, brightness/lumens, detection range, or wattage), with a reference URL for verification.
(5) Purchase Information: Confirm that the device is currently available for purchase, provide the current price in USD, and include a reference URL to a product page where the device can be purchased.
All four devices must be from different categories, and all information must be verifiable through the provided URLs.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DeviceItem(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    category_url: Optional[str] = None

    matter_url: Optional[str] = None

    ecosystem: Optional[str] = None
    ecosystem_url: Optional[str] = None

    connectivity_protocol: Optional[str] = None
    connectivity_url: Optional[str] = None

    additional_spec_name: Optional[str] = None
    additional_spec_value: Optional[str] = None
    additional_spec_url: Optional[str] = None

    price_usd: Optional[str] = None
    purchase_url: Optional[str] = None


class DevicesExtraction(BaseModel):
    devices: List[DeviceItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_devices() -> str:
    return """
    Extract up to six Matter-compatible smart home devices mentioned in the answer. For each device, return an object with the following fields:
    - name: The product name as stated in the answer.
    - category: The smart home device category as stated (e.g., smart lighting, smart lock, sensor, smart plug, switch, thermostat, camera). Use the exact wording from the answer when possible.
    - category_url: A URL (manufacturer or retailer page, or credible review/spec page) that can verify the device’s category. If multiple are provided, choose the most authoritative one.
    - matter_url: A URL that explicitly indicates Matter compatibility or certification (manufacturer, CSA certification page, or reputable retailer page where "Matter" is clearly stated).
    - ecosystem: One major ecosystem stated (normalize if possible to one of: "Apple Home", "Google Home", or "Amazon Alexa"). If the answer says "HomeKit" or "Apple HomeKit", map it to "Apple Home". If it says "Google Assistant", map to "Google Home". If it says "Alexa" or "Works with Alexa", map to "Amazon Alexa". If ambiguous, use the wording from the answer.
    - ecosystem_url: A URL that explicitly documents compatibility with the chosen ecosystem.
    - connectivity_protocol: The connectivity protocol as stated (e.g., "Thread", "Wi-Fi", "Bluetooth", "Ethernet"; allow variants like "WiFi", "IEEE 802.11", "2.4 GHz Wi‑Fi").
    - connectivity_url: A URL that documents the connectivity protocol information.
    - additional_spec_name: The name of one additional technical spec (e.g., "dimensions", "power", "brightness", "detection range", "wattage").
    - additional_spec_value: The value corresponding to that additional spec exactly as presented.
    - additional_spec_url: A URL that documents the additional spec information.
    - price_usd: The current price in USD as stated (e.g., "$24.99", "USD 24.99", "24.99 USD"). Extract exactly as written in the answer if present.
    - purchase_url: A direct product page URL where the device can be purchased (manufacturer store or reputable retailer page).

    Rules:
    - Only extract information explicitly present in the answer text.
    - For URL fields, extract actual URLs present in the answer (plain or markdown link). Do not invent URLs.
    - If a field is missing, set it to null.
    - Ensure each device is an independent item in the 'devices' array.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _ordinal(n: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    return mapping.get(n, f"Device {n}")


def _normalize_ecosystem(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = name.strip().lower()
    if "homekit" in s or "apple" in s:
        return "Apple Home"
    if "google" in s or "assistant" in s:
        return "Google Home"
    if "alexa" in s or "amazon" in s:
        return "Amazon Alexa"
    return name.strip()


def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


# --------------------------------------------------------------------------- #
# Verification logic per device                                               #
# --------------------------------------------------------------------------- #
async def verify_device(
    evaluator: Evaluator,
    parent_node,
    device: DeviceItem,
    idx: int,
    prev_categories: List[str],
) -> None:
    """
    Build the verification subtree for one device and execute all checks.
    """

    ordinal_name = _ordinal(idx + 1)
    device_node = evaluator.add_parallel(
        id=f"device_{idx + 1}",
        desc=f"{ordinal_name} Matter-compatible smart home device with complete information",
        parent=parent_node,
        critical=False,
    )

    # --------------------------- Category -------------------------------- #
    if idx == 0:
        cat_desc = "Device belongs to a clearly defined smart home device category"
        cat_ident_desc = "Device category is clearly stated (e.g., smart lighting, smart lock, sensor, switch, thermostat, etc.)"
    elif idx == 1:
        cat_desc = "Device belongs to a different smart home device category than Device 1"
        cat_ident_desc = "Device category is clearly stated and different from Device 1"
    elif idx == 2:
        cat_desc = "Device belongs to a different smart home device category than Devices 1 and 2"
        cat_ident_desc = "Device category is clearly stated and different from Devices 1 and 2"
    else:
        cat_desc = "Device belongs to a different smart home device category than Devices 1, 2, and 3"
        cat_ident_desc = "Device category is clearly stated and different from Devices 1, 2, and 3"

    cat_group = evaluator.add_sequential(
        id=f"device_{idx + 1}_category",
        desc=cat_desc,
        parent=device_node,
        critical=True,
    )

    # Leaf: category identification via answer text + uniqueness check
    cat_ident_node = evaluator.add_leaf(
        id=f"device_{idx + 1}_category_identification",
        desc=cat_ident_desc,
        parent=cat_group,
        critical=True,
    )
    prev_cat_list = ", ".join(prev_categories) if prev_categories else ""
    cat_claim_parts = []
    if _non_empty(device.category):
        cat_claim_parts.append(f"The answer clearly states the device category as '{device.category}'.")
    else:
        cat_claim_parts.append("The answer clearly states the device category (non-empty).")
    if prev_categories:
        cat_claim_parts.append(
            f"This category is different from the previously used categories: {prev_cat_list}."
        )
    category_claim = " ".join(cat_claim_parts)
    await evaluator.verify(
        claim=category_claim,
        node=cat_ident_node,
        additional_instruction="Verify directly from the answer text whether the category is explicitly stated. "
                               "For the uniqueness check, compare the provided category string against the previously listed ones "
                               "case-insensitively. Allow reasonable synonyms (e.g., 'smart bulb' ~ 'smart lighting').",
    )

    # Leaf: category URL provided (existence check)
    evaluator.add_custom_node(
        result=_non_empty(device.category_url),
        id=f"device_{idx + 1}_category_url",
        desc="URL reference provided for device category verification",
        parent=cat_group,
        critical=True,
    )

    # ----------------------- Matter Compatibility ------------------------ #
    matter_group = evaluator.add_sequential(
        id=f"device_{idx + 1}_matter_compatibility",
        desc="Device is officially Matter-compatible or Matter-certified",
        parent=device_node,
        critical=True,
    )
    # Put URL existence first to gate verification
    evaluator.add_custom_node(
        result=_non_empty(device.matter_url),
        id=f"device_{idx + 1}_matter_url",
        desc="URL reference provided for Matter compatibility verification",
        parent=matter_group,
        critical=True,
    )
    matter_leaf = evaluator.add_leaf(
        id=f"device_{idx + 1}_matter_verification",
        desc="Matter compatibility is explicitly stated in manufacturer or retailer documentation",
        parent=matter_group,
        critical=True,
    )
    await evaluator.verify(
        claim="This device is Matter-compatible or Matter-certified.",
        node=matter_leaf,
        sources=device.matter_url if _non_empty(device.matter_url) else None,
        additional_instruction="Look for explicit mentions like 'Matter', 'Works with Matter', or the Matter badge. "
                               "The claim should be supported by the provided page.",
    )

    # --------------------- Ecosystem Compatibility ----------------------- #
    eco_group = evaluator.add_sequential(
        id=f"device_{idx + 1}_ecosystem_compatibility",
        desc="Device is compatible with at least one major smart home ecosystem",
        parent=device_node,
        critical=True,
    )
    # URL existence first
    evaluator.add_custom_node(
        result=_non_empty(device.ecosystem_url),
        id=f"device_{idx + 1}_ecosystem_url",
        desc="URL reference provided for ecosystem compatibility verification",
        parent=eco_group,
        critical=True,
    )
    eco_leaf = evaluator.add_leaf(
        id=f"device_{idx + 1}_ecosystem_verification",
        desc="Compatibility with Apple Home, Google Home, or Amazon Alexa is explicitly documented",
        parent=eco_group,
        critical=True,
    )
    ecosystem_normalized = _normalize_ecosystem(device.ecosystem) or (device.ecosystem or "").strip()
    eco_claim = f"The device is compatible with {ecosystem_normalized}."
    await evaluator.verify(
        claim=eco_claim,
        node=eco_leaf,
        sources=device.ecosystem_url if _non_empty(device.ecosystem_url) else None,
        additional_instruction="Accept common synonyms: Apple Home ~ HomeKit; Google Home ~ Google Assistant; "
                               "Amazon Alexa ~ Works with Alexa. The webpage should explicitly indicate compatibility.",
    )

    # ------------------------ Technical Specifications -------------------- #
    tech_group = evaluator.add_parallel(
        id=f"device_{idx + 1}_technical_specs",
        desc="Key technical specifications are provided and verifiable",
        parent=device_node,
        critical=True,
    )

    # Connectivity protocol (sequential: URL first, then stated)
    conn_group = evaluator.add_sequential(
        id=f"device_{idx + 1}_connectivity_protocol",
        desc="Connectivity protocol is specified (Thread, Wi-Fi, Bluetooth, or Ethernet)",
        parent=tech_group,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(device.connectivity_url),
        id=f"device_{idx + 1}_connectivity_url",
        desc="URL reference provided for connectivity protocol verification",
        parent=conn_group,
        critical=True,
    )
    conn_leaf = evaluator.add_leaf(
        id=f"device_{idx + 1}_connectivity_stated",
        desc="Connectivity protocol is clearly stated",
        parent=conn_group,
        critical=True,
    )
    conn_val = (device.connectivity_protocol or "").strip()
    await evaluator.verify(
        claim=f"The device's connectivity protocol is '{conn_val}'.",
        node=conn_leaf,
        sources=device.connectivity_url if _non_empty(device.connectivity_url) else None,
        additional_instruction="Accept reasonable variants: Wi‑Fi/WiFi/IEEE 802.11/2.4 GHz Wi‑Fi, Thread, Bluetooth, Ethernet. "
                               "Verify the statement appears on the provided page.",
    )

    # Additional specification (sequential: URL first, then provided)
    spec_group = evaluator.add_sequential(
        id=f"device_{idx + 1}_additional_specs",
        desc="At least one additional technical specification is provided and documented",
        parent=tech_group,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(device.additional_spec_url),
        id=f"device_{idx + 1}_specs_url",
        desc="URL reference provided for additional specifications verification",
        parent=spec_group,
        critical=True,
    )
    spec_leaf = evaluator.add_leaf(
        id=f"device_{idx + 1}_specs_provided",
        desc="At least one additional specification is provided (power requirements, dimensions, brightness, detection range, wattage, etc.)",
        parent=spec_group,
        critical=True,
    )
    spec_name = (device.additional_spec_name or "").strip()
    spec_val = (device.additional_spec_value or "").strip()
    await evaluator.verify(
        claim=f"The device has '{spec_name}': '{spec_val}'.",
        node=spec_leaf,
        sources=device.additional_spec_url if _non_empty(device.additional_spec_url) else None,
        additional_instruction="Verify that this specific spec name and value (or very close phrasing/units) appear on the provided page.",
    )

    # ------------------------ Purchase Information ------------------------ #
    purchase_group = evaluator.add_parallel(
        id=f"device_{idx + 1}_purchase_info",
        desc="Device is currently available for purchase with pricing information and verifiable URL",
        parent=device_node,
        critical=True,
    )

    # Ensure purchase URL presence first (to gate siblings)
    evaluator.add_custom_node(
        result=_non_empty(device.purchase_url),
        id=f"device_{idx + 1}_purchase_url",
        desc="URL reference provided to product page for purchase information verification",
        parent=purchase_group,
        critical=True,
    )

    # Availability verification
    available_leaf = evaluator.add_leaf(
        id=f"device_{idx + 1}_available_verification",
        desc="Product page confirms device is currently available for purchase",
        parent=purchase_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The product page indicates the item is currently available for purchase (e.g., 'In Stock', 'Add to Cart', 'Buy now', available for order).",
        node=available_leaf,
        sources=device.purchase_url if _non_empty(device.purchase_url) else None,
        additional_instruction="Consider indicators such as 'In Stock', 'Add to Cart', 'Buy now', or explicit availability text. "
                               "If the page clearly shows 'Out of Stock', 'Pre-order', or 'Unavailable', it should be considered not available.",
    )

    # Price verification
    price_leaf = evaluator.add_leaf(
        id=f"device_{idx + 1}_price_stated",
        desc="Current price is clearly stated in USD",
        parent=purchase_group,
        critical=True,
    )
    price_str = (device.price_usd or "").strip()
    await evaluator.verify(
        claim=f"The current price is stated as '{price_str}' in USD.",
        node=price_leaf,
        sources=device.purchase_url if _non_empty(device.purchase_url) else None,
        additional_instruction="Treat the '$' symbol as USD. If multiple variants/prices exist, any clearly labeled base price in USD is acceptable. "
                               "The page should display this price or an obviously equivalent USD price.",
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
    Evaluate an answer for the Matter-compatible devices task.
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
        default_model=model,
    )

    # Extract devices
    extracted = await evaluator.extract(
        prompt=prompt_extract_devices(),
        template_class=DevicesExtraction,
        extraction_name="devices_extraction",
    )

    # Keep only the first four devices (pad if fewer)
    devices: List[DeviceItem] = list(extracted.devices[:4])
    while len(devices) < 4:
        devices.append(DeviceItem())

    # Verify each device subtree
    prev_categories: List[str] = []
    for i in range(4):
        await verify_device(
            evaluator=evaluator,
            parent_node=root,
            device=devices[i],
            idx=i,
            prev_categories=prev_categories.copy(),
        )
        # Track category for uniqueness checks (store a normalized token)
        if _non_empty(devices[i].category):
            prev_categories.append(devices[i].category.strip())

    return evaluator.get_summary()